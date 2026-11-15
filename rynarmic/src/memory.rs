//! The guest memory interface.
//!
//! dynarmic hands memory access to the host through its `MemoryReadCode`/`MemoryRead`/`MemoryWrite`
//! callbacks; the Rust version expresses the same role as a trait, so a host can supply an MMU, a
//! page table, or a direct mapping.

/// A flat window mapping a range of guest addresses onto a host pointer.
///
/// The JIT uses it to inline an access as a single host `mov`: subtract `base`, compare
/// unsigned against `len`, and on a hit touch `ptr` directly; a miss falls back to the [`Memory`]
/// callbacks. **No signal handling and no page flipping are involved.**
///
/// # Contract (the host must hold it, or the JIT reads the wrong thing)
///
/// - `ptr` points at `len` bytes corresponding to guest `[base, base + len)`, byte for byte
///   matching what [`Memory::read_u8`]/[`Memory::write_u8`] would do;
/// - the pointer stays valid **for as long as the block lives**: once registered, that memory must
///   not be reallocated or moved;
/// - anything outside the window belongs to the callbacks; a window only covers the part that can
///   be mapped directly.
#[derive(Clone, Copy, Debug)]
pub struct FastWindow {
    /// The host pointer corresponding to guest address `base`.
    pub ptr: *mut u8,
    /// The first guest address the window covers.
    pub base: u32,
    /// The window size in bytes.
    pub len: usize,
}

// The pointer's validity is the host's promise; it takes no part in comparison semantics.
unsafe impl Send for FastWindow {}
unsafe impl Sync for FastWindow {}

/// How many window segments the JIT can inline.
///
/// Eight, because a guest address space splits into megabyte-scale segments: an image, a heap of a
/// few megabytes, and a stack easily add up to more than four — and **a segment without a window
/// pays a Rust callback on every access**. Measured, inlining costs 0.79 ns/instruction against
/// 1.88 ns/instruction through the callback (`window_vs_callback` in the frame benchmark), a factor
/// of 2.4. Each extra segment adds two compares to the hot path, which is a far smaller price than
/// letting the stack fall back to the callback.
pub const MAX_WINDOWS: usize = 8;

/// The kind of access that failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessKind {
    /// Instruction fetch: the instruction `PC` points at is the one that could not be read.
    Fetch,
    /// A data read.
    Read,
    /// A data write.
    Write,
}

/// One failed access: the guest reached an address the host cannot serve.
///
/// When the host finds it cannot serve an address in `read_*`/`write_*`, it writes non-zero into
/// the slot given to [`Memory::set_fault_flag`] and keeps the detail in its own fields for
/// [`Memory::take_fault`] to hand over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AccessFault {
    /// The address that failed.
    pub addr: u32,
    /// Fetch, read, or write.
    pub kind: AccessKind,
}

pub trait Memory {
    fn read_u8(&mut self, addr: u32) -> u8;
    fn write_u8(&mut self, addr: u32, value: u8);

    /// An optional flat window. Where it returns `Some`, the JIT inlines accesses inside the
    /// window and everything else goes through the callbacks below. **The default returns `None`,
    /// which changes no behaviour.**
    fn fast_window(&self) -> Option<FastWindow> {
        None
    }

    /// The list of inlinable windows (at most [`MAX_WINDOWS`] segments).
    ///
    /// The default wraps [`Memory::fast_window`] as a single segment, so a host that only
    /// implements the older single-window method keeps working; a host that wants several segments
    /// (image, stack and heap each contiguous) overrides this method instead.
    ///
    /// **Every segment must be mapped in full**: the JIT checks only the bounds when inlining, not
    /// whether a page exists, so handing over a region with holes in it silently swallows the
    /// missing page (`Fault::Unmapped` stops happening).
    ///
    /// **Segment 0 is the primary window**: its base lives in a host register, and the constant
    /// addresses folded out of PC-relative addressing are based on it too (saving a `mov`). When
    /// the set of windows changes (a segment added or removed, `ptr`/`base`/`len` altered),
    /// [`crate::Cpu::invalidate_code_cache`] **must** be called: compiled blocks have the old
    /// pointers baked in, and a block entered through a link jump reuses the primary base the
    /// previous block installed.
    fn fast_windows(&self) -> [Option<FastWindow>; MAX_WINDOWS] {
        let mut out = [None; MAX_WINDOWS];
        out[0] = self.fast_window();
        out
    }

    /// Hands over the slot for the access-fault flag.
    ///
    /// When the host cannot serve an address in `read_*`/`write_*`, it writes non-zero into this
    /// slot; whatever the access returns then does not matter, because the CPU stops that block or
    /// instruction as soon as the callback returns. The address and direction come out through
    /// [`Memory::take_fault`].
    ///
    /// The default does nothing — **not implementing it behaves exactly as before this feature
    /// existed** (an unservable address still returns whatever the host gives, and the CPU carries
    /// on).
    ///
    /// `Cpu` rebinds the slot before every run, because the host may move the `Cpu`; a host must
    /// **not cache** this pointer.
    fn set_fault_flag(&mut self, _slot: *mut u32) {}

    /// Takes the detail of the most recent failed access (clearing it); `None` if there was none.
    fn take_fault(&mut self) -> Option<AccessFault> {
        None
    }

    fn read_u16(&mut self, addr: u32) -> u16 {
        u16::from_le_bytes([self.read_u8(addr), self.read_u8(addr.wrapping_add(1))])
    }

    fn write_u16(&mut self, addr: u32, value: u16) {
        let bytes = value.to_le_bytes();
        self.write_u8(addr, bytes[0]);
        self.write_u8(addr.wrapping_add(1), bytes[1]);
    }

    fn read_u32(&mut self, addr: u32) -> u32 {
        u32::from_le_bytes([
            self.read_u8(addr),
            self.read_u8(addr.wrapping_add(1)),
            self.read_u8(addr.wrapping_add(2)),
            self.read_u8(addr.wrapping_add(3)),
        ])
    }

    fn write_u32(&mut self, addr: u32, value: u32) {
        let bytes = value.to_le_bytes();
        for (i, byte) in bytes.iter().enumerate() {
            self.write_u8(addr.wrapping_add(i as u32), *byte);
        }
    }
}

/// The flat memory used by tests and examples: reads outside `[base, base + len)` give 0 and
/// writes are dropped.
#[derive(Clone, Debug)]
pub struct SimpleMemory {
    base: u32,
    data: Vec<u8>,
}

impl SimpleMemory {
    pub fn new(base: u32, len: usize) -> Self {
        Self {
            base,
            data: vec![0; len],
        }
    }

    /// A direct read for test assertions (out of range gives 0).
    pub fn peek_u32(&self, addr: u32) -> u32 {
        let offset = addr.wrapping_sub(self.base) as usize;
        if offset + 4 <= self.data.len() {
            u32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ])
        } else {
            0
        }
    }

    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        for (i, byte) in bytes.iter().enumerate() {
            self.write_u8(addr.wrapping_add(i as u32), *byte);
        }
    }

    pub fn write_words(&mut self, addr: u32, words: &[u32]) {
        for (i, word) in words.iter().enumerate() {
            self.write_u32(addr.wrapping_add((i as u32) * 4), *word);
        }
    }

    pub fn write_thumb(&mut self, addr: u32, halfwords: &[u16]) {
        for (i, hw) in halfwords.iter().enumerate() {
            self.write_u16(addr.wrapping_add((i as u32) * 2), *hw);
        }
    }

    fn index(&self, addr: u32) -> Option<usize> {
        let offset = addr.wrapping_sub(self.base) as usize;
        (offset < self.data.len()).then_some(offset)
    }
}

impl Memory for SimpleMemory {
    fn fast_window(&self) -> Option<FastWindow> {
        // `data` is never grown or shrunk after construction, so the buffer address is stable;
        // a host must not swap it out
        Some(FastWindow {
            ptr: self.data.as_ptr() as *mut u8,
            base: self.base,
            len: self.data.len(),
        })
    }

    fn read_u8(&mut self, addr: u32) -> u8 {
        self.index(addr).map_or(0, |i| self.data[i])
    }

    fn write_u8(&mut self, addr: u32, value: u8) {
        if let Some(i) = self.index(addr) {
            self.data[i] = value;
        }
    }

    // Slicing by width directly, to avoid the per-byte callback cost of the default methods
    fn read_u16(&mut self, addr: u32) -> u16 {
        let offset = addr.wrapping_sub(self.base) as usize;
        match self.data.get(offset..offset + 2) {
            Some(bytes) => u16::from_le_bytes([bytes[0], bytes[1]]),
            None => 0,
        }
    }

    fn write_u16(&mut self, addr: u32, value: u16) {
        let offset = addr.wrapping_sub(self.base) as usize;
        if let Some(slot) = self.data.get_mut(offset..offset + 2) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }

    fn read_u32(&mut self, addr: u32) -> u32 {
        let offset = addr.wrapping_sub(self.base) as usize;
        match self.data.get(offset..offset + 4) {
            Some(bytes) => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            None => 0,
        }
    }

    fn write_u32(&mut self, addr: u32, value: u32) {
        let offset = addr.wrapping_sub(self.base) as usize;
        if let Some(slot) = self.data.get_mut(offset..offset + 4) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
}
