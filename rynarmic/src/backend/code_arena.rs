//! A self-managed block of executable memory, appended to block by block.
//!
//! # Why it exists
//!
//! Giving every block its own `dynasmrt::Assembler::new()` (one `mmap`) plus a `finalize()`
//! (`mprotect` and a copy) measured **35.2 µs**. Assembling the same block with `VecAssembler` into
//! ordinary heap memory and copying it into a pre-reserved executable region costs **0.82 µs** — a
//! factor of 43, and that overhead was 62% of the 52 µs a single block costs to compile.
//!
//! # Why code may be moved freely
//!
//! There is **no** `rel32` link between blocks: every cross-block edge goes through a
//! [`LinkSlot`](super::x64::LinkSlot) indirect jump, so a block's code is valid at any address.
//! In-block labels (prologue, `fallthrough`, a self-loop's `body`) are relative jumps and hold as
//! long as the block moves whole — and since `VecAssembler` needs **the address the code will be
//! loaded at**, a position is requested from the arena before assembling and that address is used.
//!
//! # The cost
//!
//! This memory is **RWX**: writable, readable and executable. That is not an oversight — JIT code
//! has to be both, and flipping permissions per block is exactly the system call being avoided. The
//! price is that anything able to write this memory can change code that is running, so it is used
//! only for the host's own generated code and guest data must never be allowed to land in it.

/// Each block start is 16-byte aligned: x86 fetches fine unaligned, but aligned block starts are
/// easier to read in a disassembly and in a profile, and the cost is under one byte per block.
const BLOCK_ALIGN: usize = 16;

/// A process normally has one `CodeArena`, reserved at `MAX_ARENA_BYTES`.
///
/// 65536 cached blocks × about 170 bytes each is roughly 11 MiB, so 32 MiB leaves a factor of two
/// in hand. It is virtual address space only — the operating system commits physical pages when
/// they are actually written.
pub const MAX_ARENA_BYTES: usize = 32 << 20;

/// The minimum free space required before compiling a block. A block rarely exceeds 4 KiB (even a
/// 32-instruction block is a few hundred bytes), and 8 KiB is left so that "the last block just
/// fits" never ends up half written.
pub const MIN_BLOCK_ROOM: usize = 8 << 10;

pub struct CodeArena {
    /// The start address. Stored as a `usize` rather than `*mut u8` so that the arena stays `Send`:
    /// a `Cpu` may be moved to another thread, and a raw pointer would make it `!Send`.
    base: usize,
    /// The number of bytes reserved.
    len: usize,
    /// The number of bytes used so far, padding included. Only grows; `reset` puts it back to zero.
    used: usize,
}

impl CodeArena {
    /// Reserves `len` bytes of executable memory, rounded up to a page.
    ///
    /// A failed reservation yields an arena of capacity zero: `remaining()` is always 0, so `compile`
    /// never gets space and every block falls back to the interpreter. That beats making the
    /// constructor fallible — the interpreter is a complete, usable path, and running without a JIT
    /// is merely slow.
    pub fn new(len: usize) -> Self {
        let len = page_round(len.max(MIN_BLOCK_ROOM));
        match unsafe { reserve(len) } {
            Some(base) => Self { base, len, used: 0 },
            None => {
                eprintln!(
                    "jit: could not reserve {len} bytes of executable memory; this run is fully interpreted"
                );
                Self {
                    base: 0,
                    len: 0,
                    used: 0,
                }
            }
        }
    }

    /// Whether the memory was really obtained.
    pub fn is_available(&self) -> bool {
        self.len != 0
    }

    /// How many bytes are still allocatable.
    pub fn remaining(&self) -> usize {
        self.len.saturating_sub(self.used)
    }

    /// How many bytes have been used.
    pub fn used(&self) -> usize {
        self.used
    }

    /// The total capacity.
    pub fn capacity(&self) -> usize {
        self.len
    }

    /// Where the next block will land. This has to be asked before assembling, because
    /// `VecAssembler` needs the address the code will be loaded at to resolve in-block labels.
    pub fn next_address(&self) -> usize {
        self.base + align_up(self.used)
    }

    /// Copies `bytes` into the next position and returns its entry address.
    ///
    /// `None` when there is not enough room; a caller should check [`Self::remaining`] **before**
    /// compiling, so that no assembly is wasted.
    pub fn commit(&mut self, bytes: &[u8]) -> Option<usize> {
        let start = align_up(self.used);
        let end = start.checked_add(bytes.len())?;
        if end > self.len {
            return None;
        }
        // SAFETY: `start..end` lies inside this arena's reservation (just checked) and that range is
        // RWX, hence writable; `bytes` does not overlap it (one is on the host heap, the other in
        // this mapping).
        unsafe {
            let dst = (self.base as *mut u8).add(start);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        self.used = end;
        Some(self.base + start)
    }

    /// Discards every allocated block. **The caller must clear all `JitBlock`s at the same time**:
    /// after a reset their entry pointers aim at positions that will be overwritten.
    ///
    /// Nothing is zeroed: a new block overwrites its own stretch, and the padding between blocks is
    /// never executed.
    pub fn reset(&mut self) {
        self.used = 0;
    }
}

impl Drop for CodeArena {
    fn drop(&mut self) {
        if self.base != 0 {
            // SAFETY: `base` and `len` came from `reserve`, and this runs once.
            unsafe { release(self.base, self.len) };
        }
    }
}

impl std::fmt::Debug for CodeArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeArena")
            .field("used", &self.used)
            .field("capacity", &self.len)
            .finish()
    }
}

#[inline]
fn align_up(n: usize) -> usize {
    (n + BLOCK_ALIGN - 1) & !(BLOCK_ALIGN - 1)
}

#[inline]
fn page_round(n: usize) -> usize {
    const PAGE: usize = 4096;
    (n + PAGE - 1) & !(PAGE - 1)
}

// ---------------------------------------------------------------------------
// Platform: reserve one block of RWX memory.
//
// Only two primitives are needed, so neither libc nor windows-sys is pulled in: the constants and
// signatures here are part of a stable ABI, whereas one more dependency would have to follow
// dynasmrt's version. Non-Windows uses `mmap`, Windows uses `VirtualAlloc`; both return a
// page-aligned address and commit physical pages lazily.
// ---------------------------------------------------------------------------

#[cfg(windows)]
unsafe fn reserve(len: usize) -> Option<usize> {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;
    unsafe extern "system" {
        fn VirtualAlloc(
            address: *mut u8,
            size: usize,
            allocation_type: u32,
            protect: u32,
        ) -> *mut u8;
    }
    let at = unsafe {
        VirtualAlloc(
            std::ptr::null_mut(),
            len,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
    };
    (!at.is_null()).then_some(at as usize)
}

#[cfg(windows)]
unsafe fn release(base: usize, _len: usize) {
    const MEM_RELEASE: u32 = 0x8000;
    unsafe extern "system" {
        fn VirtualFree(address: *mut u8, size: usize, free_type: u32) -> i32;
    }
    // `MEM_RELEASE` requires a size of 0 and frees the whole reservation.
    unsafe { VirtualFree(base as *mut u8, 0, MEM_RELEASE) };
}

#[cfg(unix)]
unsafe fn reserve(len: usize) -> Option<usize> {
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 0x02;
    // Linux uses 0x20; the BSDs and macOS use 0x1000.
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    const MAP_ANON: i32 = 0x1000;
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
    const MAP_ANON: i32 = 0x20;

    unsafe extern "C" {
        fn mmap(
            address: *mut u8,
            length: usize,
            protect: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut u8;
    }
    let at = unsafe {
        mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0,
        )
    };
    // `MAP_FAILED` is -1, not null.
    (!at.is_null() && at as isize != -1).then_some(at as usize)
}

#[cfg(unix)]
unsafe fn release(base: usize, len: usize) {
    unsafe extern "C" {
        fn munmap(address: *mut u8, length: usize) -> i32;
    }
    unsafe { munmap(base as *mut u8, len) };
}

#[cfg(not(any(windows, unix)))]
unsafe fn reserve(_len: usize) -> Option<usize> {
    None
}

#[cfg(not(any(windows, unix)))]
unsafe fn release(_base: usize, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// This memory really has to be executable: the whole pooling scheme rests on it.
    ///
    /// The machine code is `mov eax, 42; ret`, called as a function. If the mapping's protection
    /// were wrong this would not return 42 but raise an access violation — and that is exactly the
    /// failure a unit test should catch.
    #[test]
    fn committed_bytes_are_executable() {
        let mut arena = CodeArena::new(1 << 16);
        if !arena.is_available() {
            eprintln!("skipped: this platform gives no executable memory");
            return;
        }
        let code = [0xB8u8, 42, 0, 0, 0, 0xC3];
        let at = arena.commit(&code).expect("a fresh arena has room");
        let run: extern "C" fn() -> u32 = unsafe { std::mem::transmute(at) };
        assert_eq!(run(), 42, "the machine code written in has to run");
    }

    /// The address asked for before assembling and the one commit lands at must agree: `VecAssembler`
    /// resolves in-block labels against the former, and if the two diverge every relative jump points
    /// somewhere else.
    #[test]
    fn next_address_matches_where_commit_lands() {
        let mut arena = CodeArena::new(1 << 16);
        if !arena.is_available() {
            return;
        }
        for round in 0..4u8 {
            let expected = arena.next_address();
            let code = [0x90u8; 7]; // nop
            let at = arena.commit(&code).expect("space");
            assert_eq!(
                at, expected,
                "block {round} landed elsewhere than announced"
            );
        }
    }

    /// Two blocks may not overlap, or a later one tramples an earlier one.
    #[test]
    fn consecutive_blocks_do_not_overlap() {
        let mut arena = CodeArena::new(1 << 16);
        if !arena.is_available() {
            return;
        }
        let a = arena.commit(&[0x90u8; 5]).expect("space");
        let b = arena.commit(&[0xCCu8; 5]).expect("space");
        assert!(
            b >= a + 5,
            "the second block {b:#x} landed inside the first {a:#x}+5"
        );
        assert_eq!(
            b % BLOCK_ALIGN,
            0,
            "a block start has to be {BLOCK_ALIGN}-byte aligned"
        );
    }

    /// When it does not fit, say so plainly instead of writing past the end.
    #[test]
    fn a_block_that_does_not_fit_is_refused() {
        let mut arena = CodeArena::new(MIN_BLOCK_ROOM);
        if !arena.is_available() {
            return;
        }
        // One block eats nearly the whole arena, so a second cannot fit.
        let big = vec![0x90u8; arena.capacity() - 16];
        assert!(arena.commit(&big).is_some());
        assert!(arena.commit(&[0x90u8; 64]).is_none());
    }

    /// After `reset` writing starts from the beginning again, which is the semantics
    /// `invalidate_code_cache` depends on.
    #[test]
    fn reset_rewinds_to_the_start() {
        let mut arena = CodeArena::new(1 << 16);
        if !arena.is_available() {
            return;
        }
        let first = arena.next_address();
        arena.commit(&[0x90u8; 5]).expect("space");
        assert!(arena.used() > 0);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), arena.capacity());
        assert_eq!(
            arena.next_address(),
            first,
            "a reset has to return to the same start"
        );
    }
}
