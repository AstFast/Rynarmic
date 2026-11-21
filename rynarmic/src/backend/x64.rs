//! The x86-64 backend: translates a [`BasicBlock`] directly into machine code.
//!
//! # Conventions
//!
//! Each block compiles to one function:
//!
//! ```text
//! extern "win64" fn(state: *mut ArmState, memory: *mut M) -> u32
//! ```
//!
//! The return value is the new PC (see [`JitFn`]); blocks of the exception kinds are not compiled at all.
//!
//! Host register allocation (Windows x64; under SysV only the first argument's register differs):
//!
//! | Register | Purpose | Notes |
//! |---|---|---|
//! | `RBX` | `&mut ArmState` | callee-saved under both ABIs |
//! | `RBP` | `*mut M` (guest memory) | callee-saved; saved only by blocks that access memory |
//! | `R12D` | CPSR | callee-saved; resident through the block, written back at its tail |
//! | `RAX` | Operation result | volatile |
//! | `RCX`/`RDX`/`R8`–`R11` | Scratch | volatile |
//!
//! The last three have to be **callee-saved**: an access calls a host callback (a Rust function),
//! and every volatile register is destroyed by the call.
//!
//! The guest registers r0–r15 **stay in memory** (no register caching yet).
//! That costs 1.3–1.6× against dynarmic's static register cache, but getting "translate directly and be correct" working first is the better trade.
//!
//! # Coverage
//!
//! Only blocks whose every operation is supported are compiled; otherwise [`compile`] returns `None` and the caller keeps using the interpreter.
//! That lets the backend widen its coverage step by step without touching correctness.
//!
//! Covered: arithmetic, logic, multiply and CLZ; conditional-execution guards; jumps and branches; MRS/MSR;
//! single-register accesses (with pre- and post-indexing plus writeback), and `LDM`/`STM`.
//! Not covered: `LDRD/STRD`, `SWP`, saturating arithmetic, halfword multiplies, register-specified shifts, exceptions — a block containing one of these bails out whole.

// `dynasm!`'s generated operand conversions trip useless_conversion; they are macro output, not this file's code
#![allow(clippy::useless_conversion)]
// The link slots must be boxed **one by one**: the machine code bakes each slot's heap address in as an immediate, so the moment a `Vec` reallocation moves a slot,
// the jump target becomes a wild pointer. clippy cannot see that constraint, so it is spelled out here once.
#![allow(clippy::vec_box)]

use std::mem::{offset_of, size_of};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::backend::code_arena::CodeArena;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, dynasm, x64::X64Relocation};

use crate::block::{BasicBlock, EntryFlags, flag_read_mask};
use crate::ir::{Address, AluOp, IrOp, MemSize, SatOp, ShiftType, Shifter, Value};
use crate::memory::{FastWindow, Memory};
use crate::state::{ArmState, Cpsr, LazyFlags};

// ---------------------------------------------------------------------------
// Register numbers (matching the x86 encoding)
// ---------------------------------------------------------------------------

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RBP: u8 = 5;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;
const R12: u8 = 12;

/// The host register holding `&mut ArmState`.
///
/// It has to be a register callee-saved under both ABIs: the machine code calls memory callbacks (Rust functions),
/// and every volatile register is destroyed by the call.
const STATE: u8 = RBX;
/// The host register holding `*mut M` (guest memory); it must survive callbacks too.
const MEM: u8 = RBP;

/// The register the first argument (`*mut ArmState`) arrives in.
#[cfg(windows)]
const ARG0: u8 = 1; // RCX
#[cfg(not(windows))]
const ARG0: u8 = 7; // RDI

/// The register the second argument (`*mut M`) arrives in.
#[cfg(windows)]
const ARG1: u8 = 2; // RDX
#[cfg(not(windows))]
const ARG1: u8 = 6; // RSI

const CPSR: u8 = R12;
/// The remaining budget of a link chain (a register copy of `ArmState::jit_budget`).
///
/// It has to be callee-saved under both ABIs: it must survive the whole chain, host callbacks included.
/// The chain loads it once on entry and writes it back once on exit, while each link jump inside a body costs one `sub`/`cmp`.
const BUDGET: u8 = 14; // R14
/// The host base of the guest's flat window ([`crate::memory::FastWindow::ptr`]).
///
/// It has to be callee-saved as well: inlined accesses use it as a base directly, and a block may call memory callbacks at any point.
/// Only `needs_memory` blocks load it, and the frame layout is uniform across all of them, which is what makes linking safe.
const WINDOW: u8 = 15; // R15
const RESULT: u8 = RAX;
const VAL_A: u8 = R8;
const VAL_B: u8 = R9;
const T0: u8 = RDX;
const T1: u8 = R10;
const T2: u8 = RCX;
const T3: u8 = R11;

// The machine code reads guest state at these offsets, so the layout must be flat and unpadded.
// The guest register array comes first and CPSR/SPSR are one u32 each; if anyone reorders ArmState's fields
// or makes Cpsr a wider type, the assertions below fail at compile time instead of computing the wrong thing at run time.
const _: () = {
    assert!(size_of::<Cpsr>() == 4, "CPSR has to be a single u32");
    assert!(size_of::<LazyFlags>() == 8, "LazyFlags has to be two u32s");
    assert!(
        offset_of!(ArmState, r) == 0,
        "r[16] has to be ArmState's first field"
    );
};

const CPSR_OFF: i32 = offset_of!(ArmState, cpsr) as i32;
const SPSR_OFF: i32 = offset_of!(ArmState, spsr) as i32;
/// The offset of the access-fault flag in `ArmState`; the host's memory implementation writes it non-zero from a callback.
const FAULT_OFF: i32 = offset_of!(ArmState, fault) as i32;
const EXC_OFF: i32 = offset_of!(ArmState, pending_exception) as i32;
const LAZY_RESULT_OFF: i32 =
    (offset_of!(ArmState, lazy_flags) + offset_of!(LazyFlags, result)) as i32;
const LAZY_FLAGS_OFF: i32 =
    (offset_of!(ArmState, lazy_flags) + offset_of!(LazyFlags, flags)) as i32;
const BUDGET_OFF: i32 = offset_of!(ArmState, jit_budget) as i32;
const COND_CACHE_OFF: i32 = offset_of!(ArmState, cond_cache) as i32;

const fn reg_off(index: u8) -> i32 {
    (offset_of!(ArmState, r) + (index as usize) * 4) as i32
}

/// The stack adjustment a memory callback needs: Windows requires 32 bytes of shadow space,
/// plus a 16-byte spill area of our own (`LDM/STM` keep their cursor address there).
#[cfg(windows)]
const CALL_PAD: i32 = 32;
#[cfg(not(windows))]
const CALL_PAD: i32 = 0;
/// The spill area's size.
const SPILL: i32 = 16;
/// The spill area's offset on `rsp`.
const SPILL_OFF: i32 = CALL_PAD;
/// The total stack frame size per block.
///
/// On entry `rsp % 16 == 8`; the prologue pushes **eight** callee-saved registers
/// (BUDGET/STATE/CPSR/MEM/WINDOW plus three pinned caches), which makes it `8` again,
/// so the amount subtracted must be `≡ 8 (mod 16)`.
const FRAME: i32 = CALL_PAD + SPILL + 8;

// Memory callbacks: `Cpu<M>`'s monomorphization embeds each function address as an immediate,
// so no function-pointer table has to live in the context and a call site pays no indirection.
unsafe extern "C" fn jit_read_u8<M: Memory>(memory: *mut u8, addr: u32) -> u32 {
    unsafe { (*(memory as *mut M)).read_u8(addr) as u32 }
}

unsafe extern "C" fn jit_read_u16<M: Memory>(memory: *mut u8, addr: u32) -> u32 {
    unsafe { (*(memory as *mut M)).read_u16(addr) as u32 }
}

unsafe extern "C" fn jit_read_u32<M: Memory>(memory: *mut u8, addr: u32) -> u32 {
    unsafe { (*(memory as *mut M)).read_u32(addr) }
}

unsafe extern "C" fn jit_write_u8<M: Memory>(memory: *mut u8, addr: u32, value: u32) {
    unsafe { (*(memory as *mut M)).write_u8(addr, value as u8) }
}

unsafe extern "C" fn jit_write_u16<M: Memory>(memory: *mut u8, addr: u32, value: u32) {
    unsafe { (*(memory as *mut M)).write_u16(addr, value as u16) }
}

unsafe extern "C" fn jit_write_u32<M: Memory>(memory: *mut u8, addr: u32, value: u32) {
    unsafe { (*(memory as *mut M)).write_u32(addr, value) }
}

/// The six access-callback addresses for one `M`, plus the optional flat windows.
struct MemFuncs {
    read: [usize; 3],
    write: [usize; 3],
    /// The inlinable window list; `[0]` is the primary window, whose base is resident in `WINDOW`
    /// and against which accesses to compile-time constant addresses are likewise based (see `emit_read_value`).
    windows: [Option<FastWindow>; crate::memory::MAX_WINDOWS],
}

impl MemFuncs {
    /// The primary window: the segment constant-address accesses are based on.
    #[inline]
    fn window(&self) -> Option<FastWindow> {
        self.windows[0]
    }
}

type ReadFn = unsafe extern "C" fn(*mut u8, u32) -> u32;
type WriteFn = unsafe extern "C" fn(*mut u8, u32, u32);

fn mem_funcs<M: Memory>() -> MemFuncs {
    // Go through a function pointer before usize: casting a function item straight to an integer is rejected by clippy, and this reads better
    let read: [ReadFn; 3] = [jit_read_u8::<M>, jit_read_u16::<M>, jit_read_u32::<M>];
    let write: [WriteFn; 3] = [jit_write_u8::<M>, jit_write_u16::<M>, jit_write_u32::<M>];
    MemFuncs {
        read: read.map(|f| f as usize),
        write: write.map(|f| f as usize),
        windows: [None; crate::memory::MAX_WINDOWS],
    }
}

/// A compiled block's signature: `fn(state, memory) -> new PC`.
///
/// PC comes back through the **return value** rather than being written to `state.r[15]`: the driver needs it after every block,
/// and going through memory would form a store-to-load dependency chain (about 4–5 cycles) sitting on the hot loop's critical path.
/// The caller writes it back to the state only when it actually needs it (leaving `run_cached`, switching to the interpreter, an exception).
#[cfg(windows)]
pub type JitFn = unsafe extern "win64" fn(*mut ArmState, *mut u8) -> u32;
#[cfg(not(windows))]
pub type JitFn = unsafe extern "sysv64" fn(*mut ArmState, *mut u8) -> u32;

/// One backpatchable link slot.
///
/// The slot lives in **its own heap allocation** (writable, stable address) rather than in a code page, for two hard reasons:
///
/// - an `ExecutableBuffer` is **RX** (`memmap2`'s `make_exec`), so rewriting code at run time would need a page-permission flip first;
/// - each block used its own mmap, so two blocks could sit more than rel32's ±2GB apart and a direct `jmp rel32`
///   could not be backpatched at all.
///
/// The price is that a link jump is `mov r11, slot` plus `jmp [r11]` instead of one 5-byte `jmp`.
pub struct LinkSlot {
    /// The **absolute address** of the target block's body; equal to [`LinkSlot::stub`] while unlinked.
    pub target: AtomicU64,
    /// The exit target while unlinked (this block's epilogue), restored on invalidation.
    pub stub: u64,
    /// The successor's guest address and instruction set, compared when backpatching.
    pub target_pc: u32,
    pub thumb: bool,
}

impl LinkSlot {
    /// The slot's own heap address (the machine code refers to it as an immediate).
    pub fn cell_ptr(&self) -> *const AtomicU64 {
        &self.target
    }
}

/// The **monomorphic inline cache** for an exchange branch (`bx`, `pop {pc}`, `mov pc`).
///
/// A block has at most one exchange terminator, so one slot per block suffices. The machine code references the slot's
/// absolute address as an immediate and, on a hit, jumps straight into the target's **body**, skipping prologue and epilogue — which is exactly
/// what keeps a function return from going through the driver: return-dense code would otherwise pay, every 3 or 4 instructions,
/// for restoring eight pinned registers, reloading, the Rust loop and a fresh table lookup (measured at about 28 cycles).
///
/// Invalidation: the slot points at a target's body, and the arena is discarded **whole** (`invalidate_code_cache`
/// clears every block at once), so no jump can dangle with its source alive and its target freed.
#[repr(C)]
pub struct ExchangeSlot {
    /// The cached target guest address (bit 0 cleared). The initial `u32::MAX` is odd,
    /// while a masked PC is always even, so it can never match by accident.
    pc: AtomicU32,
    /// Alignment padding so that `entry` lands on an 8-byte boundary (the machine code reads `[slot+8]`).
    _pad: u32,
    /// The absolute address of the target's body; 0 means not cached.
    entry: AtomicU64,
}

impl Default for ExchangeSlot {
    fn default() -> Self {
        Self {
            pc: AtomicU32::new(u32::MAX),
            _pad: 0,
            entry: AtomicU64::new(0),
        }
    }
}

impl ExchangeSlot {
    /// The slot's heap address (referenced as an immediate from the machine code).
    fn cell_ptr(&self) -> *const ExchangeSlot {
        self
    }

    /// Fills in `(target pc, target body)`.
    ///
    /// Write `entry` before `pc`: the reading side (machine code) compares `pc` first and reads `entry` only on a hit,
    /// so as long as the two writes are not reordered, it never sees "pc matches but entry is still the old value".
    /// x86's TSO makes stores visible in program order, so `Release` and `Relaxed` cost the same here.
    pub fn store(&self, pc: u32, entry: *const u8) {
        self.entry.store(entry as u64, Ordering::Release);
        self.pc.store(pc, Ordering::Release);
    }
}

/// The assembler. `VecAssembler` rather than `Assembler`: the latter `mmap`s a block of executable memory on every `new()`
/// and `mprotect`s it in `finalize()`, measured at 35.2 µs per block;
/// the former assembles into ordinary heap memory and `finalize()` hands back a `Vec<u8>`, which copying into
/// [`CodeArena`] turns into 0.82 µs.
///
/// The price is that `VecAssembler` must **know up front where the code will be loaded** (in-block labels resolve by address),
/// so `compile` first asks the arena for a position and then constructs it with that address.
type Asm = VecAssembler<X64Relocation>;

/// One compiled block.
pub struct JitBlock {
    /// The machine-code length in bytes. The code itself belongs to `CodeArena`; the block records only its length.
    len: usize,
    /// The link slots. Their heap addresses are immediate operands in the machine code, so they have to live until the block is discarded.
    /// The outer `Box` keeps each address stable (growing the `Vec<JitBlock>` cannot move the inner allocation).
    _slots: Box<[Box<LinkSlot>]>,
    entry: JitFn,
    /// The body entry (past the prologue): where a link jump lands.
    body: *const u8,
    start_pc: u32,
    thumb: bool,
    /// The frame layout class: only blocks of the same class may link (see `emit_link_site`).
    needs_memory: bool,
    /// How many guest instructions the block holds.
    count: u32,
    /// The inline-cache slot of an exchange terminator; `None` when the block has none.
    exchange: Option<Box<ExchangeSlot>>,
}

impl JitBlock {
    /// The machine-code length in bytes.
    ///
    /// For diagnostics: bytes per guest instruction is the steadiest measure of emission quality, since it depends neither on machine load
    /// but only on how many host instructions the backend emits per guest instruction (see `examples/code_stats.rs`).
    pub fn code_len(&self) -> usize {
        self.len
    }

    /// The address of the exchange terminator's inline-cache slot (null when the block has no exchange branch).
    #[inline]
    pub fn exchange_ptr(&self) -> *const ExchangeSlot {
        self.exchange
            .as_deref()
            .map_or(std::ptr::null(), |s| s.cell_ptr())
    }

    /// The machine-code entry address. The hot path calls it directly, avoiding one more `Option` or struct indirection.
    #[inline]
    pub fn entry_ptr(&self) -> *const u8 {
        self.entry as *const u8
    }

    /// The body entry (past the prologue). A link jump lands right here.
    #[inline]
    pub fn body_ptr(&self) -> *const u8 {
        self.body
    }

    #[inline]
    pub fn start_pc(&self) -> u32 {
        self.start_pc
    }

    #[inline]
    pub fn thumb(&self) -> bool {
        self.thumb
    }

    #[inline]
    pub fn needs_memory(&self) -> bool {
        self.needs_memory
    }

    #[inline]
    pub fn slots(&self) -> impl Iterator<Item = &LinkSlot> {
        self._slots.iter().map(|slot| slot.as_ref())
    }

    /// Runs the block, returning the new PC.
    ///
    /// The machine code only puts the new PC in the return value and **never** writes `state.r[15]`; the caller has to sync it.
    ///
    /// # Safety
    ///
    /// `state` and `memory` have to come from the `Cpu<M>` this block was compiled for, and be exclusively accessible
    /// for the duration of the call (the machine code reads and writes them through callbacks).
    #[inline]
    pub unsafe fn run<M>(&self, state: *mut ArmState, memory: *mut M) -> u32 {
        // The budget is exactly this block's instruction count, so the chain necessarily tests zero at the tail and exits,
        // leaving the low-level API's meaning as "run this one block".
        unsafe { (*state).jit_budget = self.count };
        let pc = unsafe { (self.entry)(state, memory as *mut u8) };
        // The low-level API may be called on its own; guarantee the CPSR flags are materialized on the way out
        unsafe { (*state).materialize_lazy_flags() };
        pc
    }
}

/// The `RYNARMIC_JIT_LOG` level: unset = 0 (silent), `1` = report failures only, `2` = report every block.
///
/// For diagnostics: an uncompiled block falls back to the interpreter whole (~40 MIPS against ~1400), so missing one hot block is a disaster;
/// level 2 adds **emission quality** (machine-code bytes per guest instruction), which depends on no machine load
/// and is the steadiest basis for asking why a block is expensive.
/// The environment variable is read once; sitting on the compile path then costs nothing (`env::var` takes a process lock, so it cannot be read per block).
fn jit_log_level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("RYNARMIC_JIT_LOG").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("3") => 3,
        Ok("") => 1,
        Ok(_) => 3,
        Err(_) => 0,
    })
}

/// Whether to report "this block was not compiled".
#[inline]
fn jit_log() -> bool {
    jit_log_level() >= 1
}

/// Compiles one basic block to machine code; returns `None` when it contains an unsupported operation.
///
/// The generic parameter `M` decides the actual addresses of the access callbacks: function addresses go into the machine code as immediates,
/// so the same [`BasicBlock`] compiles to different code for different `M` — as it must.
///
/// `arena` supplies where the code will be loaded: `VecAssembler` needs that address to resolve in-block
/// labels, so a position is requested from it first and that address is used to assemble. The returned `JitBlock` does **not** own
/// that memory: the `CodeArena` does, and `Core::invalidate_code_cache` clears the blocks and
/// resets the arena together, so the two never fall out of step.
pub fn compile<M: Memory>(
    block: &BasicBlock,
    thumb: bool,
    windows: [Option<FastWindow>; crate::memory::MAX_WINDOWS],
    arena: &mut CodeArena,
) -> Option<JitBlock> {
    if !block.ops.iter().all(is_supported) {
        if jit_log() {
            let bad = block
                .ops
                .iter()
                .find(|op| !is_supported(op))
                .map(|op| format!("{op:?}"))
                .unwrap_or_default();
            eprintln!(
                "jit: block {:#x} not compiled (unsupported operation {bad}, {} instructions)",
                block.start_pc, block.instruction_count
            );
        }
        return None;
    }

    // Ask for the address first, assemble second: in-block labels resolve against the final load address.
    let base = arena.next_address();
    let mut ops = Asm::new(base);
    let epilogue = ops.new_dynamic_label();
    let fallthrough = ops.new_dynamic_label();
    let body = ops.new_dynamic_label();
    // The single exit for access faults: every callback access in the block `jne`s here
    let fault = ops.new_dynamic_label();
    let mut slots: Vec<Box<LinkSlot>> = Vec::new();
    // The exchange terminator's inline-cache slot: its address is embedded in the machine code, so it is allocated up front and lives with the block.
    // Only a block ending in `JumpExchange` emits this code (see `emit_ops`); others need no slot.
    let has_exchange = matches!(block.ops.last(), Some(IrOp::JumpExchange { .. }));
    let exchange = has_exchange.then(|| Box::new(ExchangeSlot::default()));

    // A block that does not access memory neither saves `MEM` nor makes room for a callback's shadow space —
    // and that prologue and epilogue is where a small loop's block overhead mostly sits.
    let needs_memory = uses_memory(block);

    // ---- Prologue: save the callee-saved registers we clobber, load state (/memory), CPSR and the budget ----
    dynasm!(ops
        ; .arch x64
        ; push Rq(BUDGET)
        ; push Rq(STATE)
        ; push Rq(CPSR)
        ; mov Rq(STATE), Rq(ARG0)
        ; mov Rd(CPSR), [Rq(STATE) + CPSR_OFF]
        ; mov Rd(BUDGET), [Rq(STATE) + BUDGET_OFF]
    );
    // Pinned registers: loaded at the entry (a body — including one entered by a link jump — may assume them valid)
    // and stored back once in the epilogue; the memory copies may go stale while the block runs.
    for (_, host) in CACHE_REGS {
        dynasm!(ops ; push Rq(host));
    }
    for (guest, host) in CACHE_REGS {
        dynasm!(ops ; mov Rd(host), [Rq(STATE) + reg_off(guest)]);
    }
    if needs_memory {
        dynasm!(ops
            ; push Rq(MEM)
            ; mov Rq(MEM), Rq(ARG1)
            ; push Rq(WINDOW)
            // With eight callee-saved registers pushed, rsp % 16 == 8; subtracting FRAME (≡ 8 mod 16) brings it to zero
            ; sub rsp, FRAME
        );
    }

    if needs_memory {
        // The window base is embedded as an immediate, the same way as the callback addresses.
        // Every `needs_memory` block on a chain loads it at the same spot, so jumping into a body is safe.
        let base = windows[0].map_or(0, |window| window.ptr as i64);
        dynasm!(ops ; mov Rq(WINDOW), QWORD base);
    }

    // Where a link jump lands: a block jumped into directly **skips the prologue**, so this has to be exactly
    // "charge the budget, then the body". All blocks on a chain share a frame layout, and the exit runs the last block's own epilogue.
    dynasm!(ops ; => body);
    let body_off = ops.offset().0;
    dynasm!(ops ; sub Rd(BUDGET), block.instruction_count as i32);

    let mut funcs = mem_funcs::<M>();
    funcs.windows = windows;
    if !emit_ops(
        &mut ops,
        block,
        &funcs,
        epilogue,
        fallthrough,
        thumb,
        body,
        fault,
        exchange.as_deref(),
        &mut slots,
    ) {
        if jit_log() {
            eprintln!(
                "jit: block {:#x} not compiled (emitter gave up: the flag representation cannot be maintained in-block, {} instructions)",
                block.start_pc, block.instruction_count
            );
        }
        return None;
    }

    // ---- The sequential exit: PC = next_pc, and leave a link slot for this edge ----
    //
    // When the block ends in `Jump` this is dead code (the jump goes straight to the epilogue), but the slot is emitted anyway:
    // a conditional `MOV pc, ...` expands to "guard, write r15, Jump" as well,
    // and when its condition fails it **does** fall through here, so trying to decide whether the code is dead invites errors.
    dynasm!(ops
        ; => fallthrough
        ; mov Rd(RESULT), block.next_pc as i32
    );
    emit_link_site(
        &mut ops,
        epilogue,
        block.next_pc,
        thumb,
        block.start_pc,
        body,
        &mut slots,
    );

    // ---- The access-fault exit ----
    // Stop and return with the "block start PC": a block keeps no per-instruction PC (dynarmic's convention),
    // while the real address and direction sit in `Memory::take_fault` for the driver to turn into an `AccessFault`.
    dynasm!(ops
        ; => fault
        ; mov Rd(RESULT), block.start_pc as i32
        ; jmp => epilogue
    );

    // ---- Epilogue ----
    dynasm!(ops ; => epilogue);
    let epilogue_off = ops.offset().0;
    dynasm!(ops
        ; mov [Rq(STATE) + BUDGET_OFF], Rd(BUDGET)
        ; mov [Rq(STATE) + CPSR_OFF], Rd(CPSR)
    );
    // Store the pinned registers back: memory is the sole authority again
    for (guest, host) in CACHE_REGS {
        dynasm!(ops ; mov [Rq(STATE) + reg_off(guest)], Rd(host));
    }
    if needs_memory {
        dynasm!(ops
            ; add rsp, FRAME
            ; pop Rq(WINDOW)
            ; pop Rq(MEM)
        );
    }
    for &(_, host) in CACHE_REGS.iter().rev() {
        dynasm!(ops ; pop Rq(host));
    }
    dynasm!(ops
        ; pop Rq(CPSR)
        ; pop Rq(STATE)
        ; pop Rq(BUDGET)
        ; ret
    );

    let code = ops.finalize().ok()?;
    let len = code.len();
    // Land at the position just requested from the arena (`next_address` and `commit` must agree on the address,
    // or every absolute address resolved from an in-block label is wrong).
    let placed = arena.commit(&code)?;
    debug_assert_eq!(
        placed, base,
        "the arena placed the block at a different address than the one requested before assembling"
    );
    let entry: JitFn = unsafe { std::mem::transmute(placed) };
    if jit_log_level() >= 2 {
        eprintln!(
            "jit: block {:#x} {:2} insns -> {:4} bytes ({:5.1} bytes/insn, {} IR ops, {} guards, entry {:?})",
            block.start_pc,
            block.instruction_count,
            len,
            len as f64 / block.instruction_count.max(1) as f64,
            block.ops.len(),
            block
                .ops
                .iter()
                .filter(|op| matches!(op, IrOp::Cond { .. }))
                .count(),
            block.entry_flags,
        );
    }
    // The code page's base is only known at the end: initialize every slot to "unlinked, back to this block's epilogue".
    let epilogue_addr = base as u64 + epilogue_off as u64;
    for slot in &mut slots {
        slot.stub = epilogue_addr;
        slot.target.store(epilogue_addr, Ordering::Relaxed);
    }
    Some(JitBlock {
        len,
        _slots: slots.into_boxed_slice(),
        entry,
        body: (base + body_off) as *const u8,
        start_pc: block.start_pc,
        thumb,
        needs_memory,
        count: block.instruction_count,
        exchange,
    })
}

/// Leaves a backpatchable link slot at the block tail.
///
/// It emits:
///
/// ```text
/// cmp  r14d, 0            ; is there budget left
/// jle  epilogue           ; exhausted: leave with RESULT in hand and give control back to the driver
/// mov  r11, <slot address> ; an immediate; the slot was allocated before compiling, so its address is stable
/// jmp  qword [r11]         ; linked -> the target body; unlinked -> this block's epilogue
/// ```
///
/// **Frame compatibility**: a link jump skips the target's prologue, so every block on a chain must share one frame
/// (the same `needs_memory`), or the target's body would use an unsaved `MEM` or spill area, or its epilogue
/// would pop under the wrong frame layout. The driver performs that check when backpatching (`Cpu::patch_links`).
fn emit_link_site(
    ops: &mut Asm,
    epilogue: dynasmrt::DynamicLabel,
    target_pc: u32,
    thumb: bool,
    self_start_pc: u32,
    body: dynasmrt::DynamicLabel,
    slots: &mut Vec<Box<LinkSlot>>,
) {
    // The self-loop fast path: when the branch target is this block's own start (the usual shape being a `subs`/`bne` hot-loop back edge),
    // the landing point (this block's body) and the emitting site are in the same buffer, so rel32 always reaches,
    // and a direct `jmp` replaces the `mov r11, slot` + `jmp [r11]` indirect slot —
    // saving an indirect jump's fetch and BTB pressure. The budget is still charged at the body.
    if target_pc == self_start_pc {
        dynasm!(ops
            ; cmp Rd(BUDGET), 0
            ; jle => epilogue
            ; jmp => body
        );
        return;
    }
    let slot = Box::new(LinkSlot {
        target: AtomicU64::new(0),
        stub: 0,
        target_pc,
        thumb,
    });
    let addr = slot.cell_ptr() as u64;
    dynasm!(ops
        ; cmp Rd(BUDGET), 0
        ; jle => epilogue
        // The QWORD hint is required: a slot's heap address does not fit in 32 bits, and dynasm encodes imm32 by default
        ; mov Rq(T3), QWORD addr as i64
        ; jmp QWORD [Rq(T3)]
    );
    slots.push(slot);
}

/// Whether the block calls memory callbacks, which decides whether `MEM` is saved and shadow space is made.
///
/// This is **deliberately an exhaustive match rather than `matches!`**: when `IrOp` gains an operation the compiler reports
/// an uncovered arm here, forcing whoever added it to answer "does it access memory". The cost of missing the answer is not a compile error
/// but a silent crash: the prologue would not save `MEM` or make shadow space, and the first callback would trample the stack.
/// The shape this once took: `LoadDouble` added to `is_supported` while this was left behind.
fn uses_memory(block: &BasicBlock) -> bool {
    // The criterion is shared with the interpreter and the optimizer through one source of facts (`ir::facts`), so adding an access-style
    // operation means editing that one place and never a mismatch like "the JIT forgot to load the memory pointer".
    block.ops.iter().any(crate::ir::facts::may_fault)
}

/// Whether the operation is within the covered set.
fn is_supported(op: &IrOp) -> bool {
    match op {
        IrOp::Nop
        | IrOp::Cond { .. }
        | IrOp::ReadStatus { .. }
        | IrOp::Clz { .. }
        | IrOp::Mul { .. }
        | IrOp::MulLong { .. } => true,
        IrOp::Alu { a, b, .. } => value_supported(*a) && b.is_none_or(value_supported),
        // Jump and MSR operands go through emit_value too, so they have to be checked as well
        IrOp::Jump { target } | IrOp::JumpExchange { target } => value_supported(*target),
        IrOp::BranchConditional { target, .. } => value_supported(*target),
        IrOp::WriteStatus { src, .. } => value_supported(*src),
        IrOp::Load { address, .. } | IrOp::Store { address, .. } => address_supported(address),
        IrOp::LoadDouble { address, .. } | IrOp::StoreDouble { address, .. } => {
            address_supported(address)
        }
        IrOp::Swap { address, .. } => address_supported(address),
        IrOp::LoadMultiple { base, .. } | IrOp::StoreMultiple { base, .. } => *base != 15,
        // Saturating and halfword multiplies touch no memory and use no register-indirect addressing
        IrOp::Saturating { .. } | IrOp::MulHalf { .. } | IrOp::MulWordHalf { .. } => true,
        // SVC and BKPT compile too: the exception travels back to the driver through `ArmState::pending_exception`.
        // There is no catch-all arm: when an `IrOp` variant is added, the uncovered-arm error here is
        // the forced answer to "does this operation go into the JIT".
        IrOp::Swi { .. } | IrOp::Breakpoint { .. } => true,
    }
}

fn address_supported(address: &Address) -> bool {
    match address {
        Address::Absolute(_) => true,
        Address::Register { base, offset, .. } => *base != 15 && value_supported(*offset),
    }
}

fn value_supported(value: Value) -> bool {
    match value {
        Value::Reg(_) | Value::Imm(_) | Value::ImmC { .. } => true,
        // Immediates and register-specified shift amounts are both implemented
        Value::Shifted {
            by: Shifter::Imm(_),
            ..
        } => true,
        Value::Shifted {
            by: Shifter::Reg(_),
            ..
        } => true,
    }
}

/// Where the condition flags stand inside a block.
///
/// Four invariants, each of which a new operation has to be checked against:
/// - the NZCV in CPSR may only be read from [`FlagMode::Materialized`], or what comes back is the stale block-entry value
/// - before an instruction that destroys the x86 flags (including an access's host callback) [`ensure_saved`] is required,
///   or the C and V held in `X86` are lost (N and Z sit in `state.lazy.result` and survive)
/// - `emit_value`'s RRX arm reads CPSR with `bt`, so **every** position that may carry an RRX operand
///   has to materialize first; before wiring up a new IR operation, check whether its operands read C
/// - `Dead` may only be concluded from `BasicBlock::flag_live_at`: from entering it until the next full flag write
///   **nothing reads the flags**. Any read therefore has to refuse compilation here rather than read a dirty value
#[derive(Clone, Copy, PartialEq, Eq)]
enum FlagMode {
    /// The NZCV in CPSR is valid.
    Materialized,
    /// The x86 flags were just produced by an arithmetic instruction that writes all of NZCV; the result is in state.lazy.result.
    X86 { sub_like: bool },
    /// state.lazy holds a valid deferred descriptor.
    Descriptor,
    /// The flags have been destroyed, but liveness guarantees nobody reads them before the next full write.
    ///
    /// This is what lets a conditional-execution block keep using deferred flags when "the flags are dead past the merge point":
    /// the guard's two paths merge each carrying its own x86 flags, and neither will be read again.
    Dead,
    /// Neither the x86 flags nor CPSR can be trusted, but `cond`'s truth value has been captured: there is a copy
    /// in the `state.cond_cache` byte, and when `in_t3` holds, **a copy in T3's low byte too** (for cmov chains only;
    /// within a chain `emit_alu(set_flags=false)` is known not to touch T3 — every T3 touch in `emit_shift_reg`
    /// is behind `need_carry`).
    ///
    /// Entered only when "the guard region destroys the flags while the next guard reads that same condition":
    /// capture once and let each later guard test it, an order of magnitude cheaper than packing the flags into a descriptor.
    /// Any use other than "a guard on the same condition" or "a full flag write" refuses compilation.
    ///
    /// `in_t3` is set at capture; a C4.1 **branch** guard invalidates it at the merge point
    /// (the skipped path may have run code that destroys T3), while a cmov guard keeps it true throughout
    /// (there is no branch, so every path runs the same emitted sequence).
    Captured { cond: u32, in_t3: bool },
}

/// Whether the flags have already been captured as the given condition.
fn captured_is(mode: FlagMode, cond: u32) -> bool {
    matches!(mode, FlagMode::Captured { cond: captured, .. } if captured == cond)
}

/// Turns the current flags into a descriptor before an instruction that would destroy the x86 flags.
fn ensure_saved(ops: &mut Asm, mode: &mut FlagMode) {
    if let FlagMode::X86 { sub_like } = *mode {
        emit_descriptor_flags(ops, sub_like);
        *mode = FlagMode::Descriptor;
    }
}

/// Materializes the deferred flags before an instruction that needs the full CPSR flags.
///
/// `false` means the current mode cannot supply valid flags (`Dead`), and the caller should abandon compilation.
/// A read on a path that should not have one means liveness is wrong, and falling back to the interpreter is safer than emitting code that reads a dirty CPSR.
fn ensure_materialized(ops: &mut Asm, mode: &mut FlagMode) -> bool {
    match *mode {
        FlagMode::X86 { sub_like } => emit_materialize_x86(ops, sub_like),
        FlagMode::Descriptor => emit_materialize_state(ops),
        FlagMode::Materialized => {}
        // Dead means "nobody reads them" and Captured means "they are already gone"; reaching here means an upstream judgement was wrong
        FlagMode::Dead | FlagMode::Captured { .. } => return false,
    }
    *mode = FlagMode::Materialized;
    true
}

/// Preserves the flags across an instruction that would destroy them.
///
/// When nothing reads the flags after this instruction, not even the descriptor is needed: declare [`FlagMode::Dead`]
/// and liveness will prove that later guards or materializations "cannot happen".
fn save_or_kill(ops: &mut Asm, mode: &mut FlagMode, dead_after: bool) {
    if dead_after {
        *mode = FlagMode::Dead;
    } else {
        ensure_saved(ops, mode);
    }
}

/// Prepares CPSR for an instruction that **writes the flags in full**.
///
/// The only difference from [`ensure_materialized`] is `Dead`: the old flags are no longer read,
/// so materializing them is wasted work. Stale values left in CPSR can only land in bits that "this instruction does not write and nothing reads later"
/// (liveness guarantees that), so no observable wrong value escapes.
fn ensure_materialized_for_write(ops: &mut Asm, mode: &mut FlagMode) -> bool {
    if *mode == FlagMode::Dead {
        // `Materialized` must imply "the descriptor is clean": otherwise the next block would take a stale descriptor as valid
        // and overwrite the CPSR just written. Clear it here while passing.
        dynasm!(ops ; mov DWORD [Rq(STATE) + LAZY_FLAGS_OFF], 0);
        *mode = FlagMode::Materialized;
        return true;
    }
    ensure_materialized(ops, mode)
}

/// A conditional guard's merge point, before it is resolved.
#[derive(Clone, Copy)]
struct Merge {
    /// The IR position of the merge. Liveness there decides whether the flags still count as valid afterwards.
    end: usize,
    label: dynasmrt::DynamicLabel,
    /// The flag representation after both paths merge, known statically (see `IrOp::Cond` in `emit_ops`).
    mode: FlagMode,
    /// `allow_lazy` from before the guard, restored on leaving the region.
    allow_lazy: bool,
}

/// Whether a guard region leaves the current flag representation unchanged.
///
/// Only when "both paths still use one representation after merging" may `mode` carry past the guard.
/// This judgement **does not simulate emission**; it looks only at operation kinds, and anything uncertain returns `false`
/// so the caller falls back to the proven route of "materialize on entry, eager inside the region".
fn region_keeps_mode(block: &BasicBlock, from: usize, to: usize, mode: FlagMode) -> bool {
    let mut current = mode;
    for op in &block.ops[from..to] {
        // Reading the flags materializes CPSR and so changes the representation (reading them while CPSR is valid is free and counts as no change)
        if current != FlagMode::Materialized && flag_read_mask(op) != 0 {
            return false;
        }
        current = match current {
            // A guard that reads the flags (Dead) or whose flags are gone (Captured) cannot carry it over
            FlagMode::Dead | FlagMode::Captured { .. } => return false,
            // CPSR valid: as long as the region writes no flags, both entry and exit have CPSR valid
            FlagMode::Materialized => {
                if writes_nzcv(op) {
                    return false;
                }
                current
            }
            // Descriptor valid: a flag write overwrites the descriptor, and the x86 flags do not necessarily follow
            FlagMode::Descriptor => {
                if writes_nzcv(op) {
                    return false;
                }
                current
            }
            FlagMode::X86 { sub_like } => match op {
                // Deferred arithmetic of the same class: the x86 flags are replaced wholesale, so the representation holds
                IrOp::Alu {
                    op,
                    a,
                    b,
                    set_flags: true,
                    ..
                } if alu_can_be_lazy(*op, *a, *b) && alu_sub_like(*op) == sub_like => current,
                // Add and subtract without flags are emitted with LEA and likewise leave the x86 flags alone
                IrOp::Alu {
                    op: AluOp::Add | AluOp::Sub,
                    a,
                    b,
                    set_flags: false,
                    ..
                } if !value_clobbers_x86(*a)
                    && b.is_none_or(|value| !value_clobbers_x86(value)) =>
                {
                    current
                }
                // Operations that neither write the flags nor destroy the x86 ones
                IrOp::Alu {
                    op,
                    a,
                    b,
                    set_flags: false,
                    ..
                } if !alu_clobbers_x86(*op, *a, *b) => current,
                IrOp::Nop => current,
                _ => return false,
            },
        };
    }
    true
}

/// Whether the operation writes the condition flags (any of NZCV).
fn writes_nzcv(op: &IrOp) -> bool {
    match op {
        IrOp::Alu { set_flags, .. }
        | IrOp::Mul { set_flags, .. }
        | IrOp::MulLong { set_flags, .. }
        | IrOp::MulHalf { set_flags, .. }
        | IrOp::MulWordHalf { set_flags, .. } => *set_flags,
        // MSR may rewrite the flags wholesale
        IrOp::WriteStatus { spsr: false, .. } => true,
        _ => false,
    }
}

/// `Alu`'s carry-direction class: `true` means x86's CF is a borrow, so ARM's C is its opposite.
fn alu_sub_like(op: AluOp) -> bool {
    matches!(op, AluOp::Sub | AluOp::Cmp | AluOp::Rsb)
}

/// Tests the condition in the current flag representation and jumps, without changing the representation.
///
/// `false` means the current mode has no valid flags (`Dead`), and the caller should abandon compilation.
fn jump_if_mode(ops: &mut Asm, cond: u32, mode: FlagMode, label: dynasmrt::DynamicLabel) -> bool {
    match mode {
        FlagMode::X86 { sub_like } => jump_if_x86(ops, cond, sub_like, label),
        FlagMode::Descriptor => {
            jump_if_lazy(ops, cond, label);
            true
        }
        FlagMode::Materialized => {
            jump_if(ops, cond, label);
            true
        }
        FlagMode::Dead | FlagMode::Captured { .. } => false,
    }
}

/// Whether a guard region writes the condition flags.
fn region_writes_flags(block: &BasicBlock, from: usize, to: usize) -> bool {
    block.ops[from..to].iter().any(writes_nzcv)
}

/// After a capture, whether this chain is **certainly** consumed by a guard on the same condition, with the flags dead afterwards.
///
/// The test: the operation right after the region is also a `Cond` on the same condition, and once its own region ends
/// the flags are no longer read. Then `Captured` survives only as far as a guard on that condition and never leaks to where full flags are needed.
fn captured_chain_terminates(block: &BasicBlock, end: usize, cond: u32) -> bool {
    match block.ops.get(end) {
        Some(IrOp::Cond {
            cond: next_cond,
            skip,
        }) if *next_cond == cond => block.flags_dead_at(end + 1 + *skip as usize),
        _ => false,
    }
}

/// One committable write inside a guard region: `(ALU operation, destination, first operand, second operand)`.
type CmovWrite = (AluOp, u8, Value, Option<Value>);

/// Whether this conditional write can be committed with a **mask**: `dst = dst OP (b & mask)`.
///
/// The mask form requires that "OP with 0 leaves the value unchanged" (add, sub, orr, eor and bic all satisfy it)
/// and that `dst` is the first operand (the old value already lives in the destination, so nothing has to be saved).
/// Where that holds, committing becomes "`and` plus one ALU" instead of "compute the new value, `test`, `cmov`",
/// two instructions fewer and no cmov forwarding dependency on the critical path.
///
/// `adc`/`sbc`/RRX read the carry and cannot use it (a mask would break C's meaning); `mov`/`mvn` do not feed back
/// into the destination, so the mask form degrades to a three-operand select and loses to `cmov`.
fn maskable_write(op: AluOp, dst: u8, a: Value, b: Option<Value>) -> Option<Value> {
    if !matches!(
        op,
        AluOp::Add | AluOp::Sub | AluOp::Orr | AluOp::Eor | AluOp::Bic
    ) {
        return None;
    }
    if !matches!(a, Value::Reg(r) if r == dst) {
        return None;
    }
    b.filter(|b| !value_reads_carry(*b))
}

/// Emits the one ALU operation computing `target = target OP (T1 & mask)`.
///
/// `Bic` (bit clear) has no direct x86 instruction: `not T1` followed by `and`.
/// With a zero mask `T1 = 0`, which inverts to all ones and leaves `and` a no-op — exactly right.
fn emit_masked_op(ops: &mut Asm, op: AluOp, target: u8) {
    match op {
        AluOp::Add => dynasm!(ops ; add Rd(target), Rd(T1)),
        AluOp::Sub => dynasm!(ops ; sub Rd(target), Rd(T1)),
        AluOp::Orr => dynasm!(ops ; or Rd(target), Rd(T1)),
        AluOp::Eor => dynasm!(ops ; xor Rd(target), Rd(T1)),
        AluOp::Bic => dynasm!(ops
            ; not Rd(T1)
            ; and Rd(target), Rd(T1)
        ),
        _ => unreachable!("maskable_write has already filtered these"),
    }
}

/// Whether a guard region can go branchless: every operation is either a `Nop` or an ALU that writes one register,
/// writes no flags and reads no carry — so the new value can be computed **unconditionally** and committed with `cmov`.
///
/// Accesses, multiplies, flag writes and carry reads (ADC/SBC/RRX) are all out: the first has callback side effects,
/// and the rest either cannot be committed with `cmov` (the flags) or would read flags already destroyed inside the region.
fn cmov_region(region: &[IrOp]) -> Option<Vec<CmovWrite>> {
    let mut writes = Vec::new();
    for op in region {
        match *op {
            IrOp::Nop => {}
            IrOp::Alu {
                op,
                dst: Some(dst),
                a,
                b,
                set_flags: false,
            } if !op.is_test() && !alu_reads_carry(op, a, b) => {
                writes.push((op, dst, a, b));
            }
            _ => return None,
        }
    }
    if writes.is_empty() {
        None
    } else {
        Some(writes)
    }
}

/// Whether `capture_condition_x86` will succeed for this condition (`HS`/`LS` depend on sub_like; the rest always do).
fn cond_capturable(cond: u32, sub_like: bool) -> bool {
    !matches!(cond, 0x8 | 0x9) || sub_like
}

/// Emits a guard region as "compute unconditionally, commit with `cmov`" (C4.2).
///
/// Where the condition comes from: under `X86` it is captured with `setcc` first, which must happen **before** computing the values,
/// because computing them (shifted operands especially) destroys the x86 flags; under `Captured` the condition already sits in
/// `cond_cache` and is used directly. Committing is `cmp byte [cond_cache], 0` plus `cmovne`:
/// that `cmp` destroys the x86 flags, but the capture chain's admission condition guarantees nobody reads them afterwards.
///
/// Against the branch version this costs three extra instructions per write in exchange for one unpredictable `jz`;
/// a branch is cheaper when the condition is predictable, so it is used only for chains whose region destroys the flags (the caller guarantees it).
/// Captures the condition as a **0 / -1 mask** in `T3`.
///
/// `cs` and `cc` are the two commonest conditions in conditionally executed code, so their mask comes straight from CF
/// (`cmc` plus `sbb`, two instructions and 4 bytes, or one `sbb`); other conditions go through `setcc` and a reshape
/// (`setcc`, `movzx`, `neg`). A non-zero mask is equivalent to the condition holding, and that is compatible both with consuming
/// a 0/1 byte via `test` and with `cmp byte`.
///
/// The condition byte (`cond_cache`) is stored only when **some guard really does take the memory path**
/// (see [`captured_byte_needed`]): a pure cmov chain uses T3 throughout, and storing it would be a dead write.
fn capture_condition_mask(
    ops: &mut Asm,
    cond: u32,
    sub_like: bool,
    block: &BasicBlock,
    end: usize,
) {
    let from_carry = matches!(cond, 0x2 | 0x3);
    if from_carry {
        // The polarity of ARM's C against x86's CF: opposite for subtraction (C = !CF), the same for addition (C = CF).
        // mask = -(C is true), and `(cmc;) sbb T3,T3` gives exactly -CF or -(!CF).
        match (cond, sub_like) {
            (0x2, true) | (0x3, false) => dynasm!(ops ; cmc ; sbb Rd(T3), Rd(T3)),
            _ => dynasm!(ops ; sbb Rd(T3), Rd(T3)),
        }
    } else {
        capture_condition_x86(ops, cond, sub_like);
        // setcc writes only the low byte and leaves the rest as garbage; the mask is used as a whole register, so zero-extend first
        dynasm!(ops
            ; movzx Rd(T3), Rb(T3)
            ; neg Rd(T3)
        );
    }
    if captured_byte_needed(block, end, cond) {
        dynasm!(ops ; mov BYTE [Rq(STATE) + COND_CACHE_OFF], Rb(T3));
    }
}

/// Whether a later guard will read the `cond_cache` byte.
///
/// In a chain emitted by [`emit_cmov_region`], a guard on the same condition consumes it in one of two ways:
/// a cmov-committable region uses T3 (touching no memory); a region that cannot use cmov has `captured_is` read the memory byte.
/// So walking the same-condition guard chain forward from this region's end, the first guard that cannot use cmov requires the byte;
/// and if the chain is interrupted by a non-guard operation (usually the next full flag write) the mode becomes `Dead` and nobody reads the byte.
fn captured_byte_needed(block: &BasicBlock, mut cursor: usize, cond: u32) -> bool {
    loop {
        match block.ops.get(cursor) {
            Some(IrOp::Cond { cond: c, skip }) if *c == cond => {
                let end = cursor + 1 + *skip as usize;
                if cmov_region(&block.ops[cursor + 1..end]).is_none() {
                    return true;
                }
                cursor = end;
            }
            _ => return false,
        }
    }
}

fn emit_cmov_region(
    ops: &mut Asm,
    block: &BasicBlock,
    end: usize,
    mode: &mut FlagMode,
    cond: u32,
    writes: &[CmovWrite],
) {
    // The condition is usable straight from T3 (a register test, one cycle, no memory):
    // true right after a capture, and a preceding cmov region keeps it true (see `Captured::in_t3`).
    // Only a Captured that came through a C4.1 branch guard degrades to reading the memory byte.
    //
    // Negate right after capturing (0/1 -> 0/-1): T3 goes from a "condition byte" to a **mask**, which the feedback form
    // of committing (see `maskable_write`) uses for the bitwise and; the non-feedback form still does
    // `test T3` plus `cmovnz` — a non-zero mask is equivalent to the condition holding, so both uses mean the same.
    let in_t3 = match *mode {
        FlagMode::X86 { sub_like } => {
            // Captured as a 0/-1 mask in T3; the `cond_cache` byte is stored only when a memory-path consumer exists.
            capture_condition_mask(ops, cond, sub_like, block, end);
            true
        }
        FlagMode::Captured { in_t3, .. } => in_t3,
        _ => unreachable!("the caller guarantees the mode is X86 or a same-condition Captured"),
    };
    for (op, dst, a, b) in writes {
        // The feedback form (dst is the first operand) prefers mask committing: one cmov fewer, one fewer copy of
        // the old value, and it replaces a cmov on the critical path with an ordinary ALU.
        //
        // It feeds the condition in T3, so it is only usable while T3 still holds the mask. A `Captured`
        // that came through a branch guard has no T3 copy — the skipped path may clobber it — and the
        // memory byte is 0/1 (or the low byte of the mask), not a mask; that case takes the cmov route
        // below, which reads `cond_cache` instead.
        let maskable = if in_t3 {
            maskable_write(*op, *dst, *a, *b)
        } else {
            None
        };
        if let Some(operand) = maskable {
            // T1 = b & mask (with b = 0 the result is unchanged, which is precisely "the condition fails")
            emit_value(ops, operand, T1, false);
            dynasm!(ops ; and Rd(T1), Rd(T3));
            match cache_of(*dst) {
                Some(host) => emit_masked_op(ops, *op, host),
                None => {
                    load_guest(ops, T0, *dst);
                    emit_masked_op(ops, *op, T0);
                    store_guest(ops, *dst, T0);
                }
            }
            continue;
        }
        // The new value goes into RESULT; `dst=None` makes emit_alu skip the usual store-back
        emit_alu(ops, *op, None, *a, *b, false, false, false);
        // With the destination already pinned in a host register, commit with `cmov` directly (relaying through T1 costs two more copies,
        // and conditional execution is a large share of real code — one hot block of `kern_div` has eight of them).
        match cache_of(*dst) {
            Some(host) if host != RESULT => {
                if in_t3 {
                    dynasm!(ops
                        ; test Rb(T3), Rb(T3)
                        ; cmovnz Rd(host), Rd(RESULT)
                    );
                } else {
                    dynasm!(ops
                        ; cmp BYTE [Rq(STATE) + COND_CACHE_OFF], 0
                        ; cmovne Rd(host), Rd(RESULT)
                    );
                }
            }
            _ => {
                load_guest(ops, T1, *dst);
                if in_t3 {
                    dynasm!(ops
                        ; test Rb(T3), Rb(T3)
                        ; cmovnz Rd(T1), Rd(RESULT)
                    );
                } else {
                    dynasm!(ops
                        ; cmp BYTE [Rq(STATE) + COND_CACHE_OFF], 0
                        ; cmovne Rd(T1), Rd(RESULT)
                    );
                }
                store_guest(ops, *dst, T1);
            }
        }
    }
    // The commit has destroyed the x86 flags; the condition lives on in T3 (and its memory copy).
    // A cmov chain never branches and nothing destroys T3 along the way, so in_t3 stays true.
    // If another same-condition guard follows, let it be consumed there; otherwise drop to Dead (nobody can consume it).
    *mode = match block.ops.get(end) {
        Some(IrOp::Cond {
            cond: next,
            skip: _,
        }) if *next == cond => FlagMode::Captured { cond, in_t3 },
        _ => FlagMode::Dead,
    };
}

/// Emits every operation in the block. `false` means an operation could not be encoded and compilation should be abandoned.
#[allow(clippy::too_many_arguments)]
fn emit_ops(
    ops: &mut Asm,
    block: &BasicBlock,
    funcs: &MemFuncs,
    epilogue: dynasmrt::DynamicLabel,
    fallthrough: dynasmrt::DynamicLabel,
    thumb: bool,
    body: dynasmrt::DynamicLabel,
    fault: dynasmrt::DynamicLabel,
    exchange: Option<&ExchangeSlot>,
    slots: &mut Vec<Box<LinkSlot>>,
) -> bool {
    if block.entry_flags == EntryFlags::NeedsMaterialized {
        emit_materialize_state(ops);
    }
    let mut mode = FlagMode::Materialized;
    if block.entry_flags == EntryFlags::Discards {
        // Discard the descriptor the previous block left; the first full write rebuilds the flags
        dynasm!(ops ; mov DWORD [Rq(STATE) + LAZY_FLAGS_OFF], 0);
    }
    let mut allow_lazy = true;
    let mut pending: Vec<Merge> = Vec::new();

    let mut index = 0;
    while index < block.ops.len() {
        let trace = jit_log_level() >= 3;
        let offset_before = if trace { ops.offset().0 } else { 0 };
        while let Some(merge) = pending.last() {
            if merge.end != index {
                break;
            }
            let Merge {
                label,
                mode: merged,
                allow_lazy: lazy,
                ..
            } = *merge;
            dynasm!(ops ; => label);
            mode = merged;
            allow_lazy = lazy;
            pending.pop();
        }

        let op = block.ops[index];
        // Whether the flags are unread after this operation: if so, even preserving them is wasted work
        let dead_after = block.flags_dead_at(index + 1);
        match op {
            IrOp::Nop => {}
            IrOp::Cond { cond, skip } => {
                let end = index + 1 + skip as usize;

                // C4.2: branchless. When a guard region holds only ALUs that write a register and leave the flags alone,
                // "skip a stretch" becomes "compute unconditionally, commit with cmov" — no misprediction,
                // and one captured condition can feed any number of commits. Under X86 the chain must first be confirmed to be
                // consumed by same-condition guards (the same admission line as the C4.1 capture).
                let cmov_writes = match mode {
                    FlagMode::X86 { sub_like }
                        if cond_capturable(cond, sub_like)
                            && captured_chain_terminates(block, end, cond) =>
                    {
                        cmov_region(&block.ops[index + 1..end])
                    }
                    FlagMode::Captured { cond: captured, .. } if captured == cond => {
                        cmov_region(&block.ops[index + 1..end])
                    }
                    _ => None,
                };
                if let Some(writes) = cmov_writes {
                    emit_cmov_region(ops, block, end, &mut mode, cond, &writes);
                    index = end;
                    continue;
                }

                let label = ops.new_dynamic_label();

                // The same condition was already captured: reading that byte (two instructions) suffices,
                // with no need to touch the flags — which is exactly the way out when a region has a shift and the flags are gone.
                if captured_is(mode, cond) {
                    if !block.flags_dead_at(end) {
                        // The chain check says this cannot happen; if it does, abandon compilation
                        return false;
                    }
                    dynasm!(ops
                        ; cmp BYTE [Rq(STATE) + COND_CACHE_OFF], 0
                        ; jz => label
                    );
                    pending.push(Merge {
                        end,
                        label,
                        mode: FlagMode::Dead,
                        allow_lazy,
                    });
                    index += 1;
                    continue;
                }
                // If the flags are dead past the merge point, the two paths need not keep "one representation":
                // the skipped side keeps the x86 flags from before the guard, the executed side keeps the ones it just computed,
                // and neither will be read. `__udivsi3`'s `cmp; subcs; orrcs` has exactly this shape.
                let dead_merge = block.flags_dead_at(end);
                if dead_merge || region_keeps_mode(block, index + 1, end, mode) {
                    if !jump_if_mode(ops, cond ^ 1, mode, label) {
                        return false;
                    }
                    let merged = if dead_merge { FlagMode::Dead } else { mode };
                    pending.push(Merge {
                        end,
                        label,
                        mode: merged,
                        allow_lazy,
                    });
                } else if matches!(mode, FlagMode::X86 { .. })
                    && !region_writes_flags(block, index + 1, end)
                    && captured_chain_terminates(block, end, cond)
                {
                    // The region writes no flags but destroys the x86 ones (a shifted `subcs` being typical),
                    // while the same set of flags is read after the merge: store the condition as one byte with `setcc`
                    // so later same-condition guards read it directly, saving the 9 to 13 instructions of flag packing.
                    let FlagMode::X86 { sub_like } = mode else {
                        unreachable!("just decided above")
                    };
                    if !capture_condition_x86(ops, cond, sub_like) {
                        return false;
                    }
                    dynasm!(ops
                        ; mov BYTE [Rq(STATE) + COND_CACHE_OFF], Rb(T3)
                        ; test Rb(T3), Rb(T3)
                        ; jz => label
                    );
                    pending.push(Merge {
                        end,
                        label,
                        // The skipped path may have run code that destroys T3, so past the merge only the memory copy is trustworthy
                        mode: FlagMode::Captured { cond, in_t3: false },
                        allow_lazy,
                    });
                } else {
                    // The region writes the flags and they are read after the merge: fall back to "materialize on entry, eager inside the region",
                    // so both paths merge with an equally valid CPSR.
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                    jump_if(ops, cond ^ 1, label);
                    pending.push(Merge {
                        end,
                        label,
                        mode: FlagMode::Materialized,
                        allow_lazy,
                    });
                    allow_lazy = false;
                }
            }
            IrOp::Alu {
                op,
                dst,
                a,
                b,
                set_flags,
            } => {
                let reads_carry = alu_reads_carry(op, a, b);
                let use_lazy = set_flags && allow_lazy && alu_can_be_lazy(op, a, b);
                // Add and subtract without flags go through LEA and leave the x86 flags alone.
                // A conditional-execution block's `cmp; subcs; orrcs` relies on that: both guards read one set of flags,
                // and if the `subcs` between them destroyed the flags the only option would be to materialize first (13 instructions).
                let keep_flags = !set_flags
                    && !dead_after
                    && matches!(mode, FlagMode::X86 { .. })
                    && matches!(op, AluOp::Add | AluOp::Sub)
                    && !value_clobbers_x86(a)
                    && b.is_none_or(|value| !value_clobbers_x86(value));
                if reads_carry {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else if set_flags && !use_lazy {
                    if !ensure_materialized_for_write(ops, &mut mode) {
                        return false;
                    }
                } else if !keep_flags && !set_flags && alu_clobbers_x86(op, a, b) {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_alu(ops, op, dst, a, b, set_flags, use_lazy, keep_flags);
                if set_flags {
                    if use_lazy {
                        emit_lazy_result_store(ops);
                        mode = FlagMode::X86 {
                            sub_like: matches!(op, AluOp::Sub | AluOp::Cmp | AluOp::Rsb),
                        };
                    } else {
                        mode = FlagMode::Materialized;
                    }
                }
            }
            IrOp::Mul {
                dst,
                a,
                b,
                accumulate,
                set_flags,
            } => {
                if set_flags {
                    if !ensure_materialized_for_write(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_mul(ops, dst, a, b, accumulate, set_flags);
                if set_flags {
                    mode = FlagMode::Materialized;
                }
            }
            IrOp::MulLong {
                lo,
                hi,
                a,
                b,
                signed,
                accumulate,
                set_flags,
            } => {
                if set_flags {
                    if !ensure_materialized_for_write(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_mul_long(ops, lo, hi, a, b, signed, accumulate, set_flags);
                if set_flags {
                    mode = FlagMode::Materialized;
                }
            }
            IrOp::Clz { dst, src } => {
                save_or_kill(ops, &mut mode, dead_after);
                emit_clz(ops, dst, src);
            }
            IrOp::ReadStatus { dst, spsr } => {
                if !ensure_materialized(ops, &mut mode) {
                    return false;
                }
                if spsr {
                    dynasm!(ops ; mov Rd(RESULT), [Rq(STATE) + SPSR_OFF]);
                } else {
                    dynasm!(ops ; mov Rd(RESULT), Rd(CPSR));
                }
                store_guest(ops, dst, RESULT);
                mode = FlagMode::Materialized;
            }
            IrOp::WriteStatus { src, mask, spsr } => {
                if !ensure_materialized_for_write(ops, &mut mode) {
                    return false;
                }
                emit_write_status(ops, src, mask, spsr);
                mode = FlagMode::Materialized;
            }
            IrOp::Load {
                dst,
                address,
                size,
                signed,
                pre_index,
                writeback,
            } => {
                if address_reads_carry(&address) {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_load(
                    ops, funcs, fault, dst, address, size, signed, pre_index, writeback,
                );
            }
            IrOp::Store {
                src,
                address,
                size,
                pre_index,
                writeback,
            } => {
                if address_reads_carry(&address) {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_store(ops, funcs, fault, src, address, size, pre_index, writeback);
            }
            IrOp::LoadMultiple {
                base,
                regs,
                add,
                pre_index,
                writeback,
            } => {
                save_or_kill(ops, &mut mode, dead_after);
                emit_load_multiple(ops, funcs, fault, base, regs, add, pre_index, writeback);
            }
            IrOp::LoadDouble {
                rt,
                address,
                pre_index,
                writeback,
            } => {
                if address_reads_carry(&address) {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_load_double(ops, funcs, fault, rt, address, pre_index, writeback);
            }
            IrOp::StoreDouble {
                rt,
                address,
                pre_index,
                writeback,
            } => {
                if address_reads_carry(&address) {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_store_double(ops, funcs, fault, rt, address, pre_index, writeback);
            }
            IrOp::Swap {
                dst,
                src,
                address,
                byte,
            } => {
                if address_reads_carry(&address) {
                    if !ensure_materialized(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_swap(ops, funcs, fault, dst, src, address, byte);
            }
            // Saturating arithmetic writes only CPSR.Q, touches no NZCV and needs no prior materialization
            IrOp::Saturating { op, dst, a, b } => emit_saturating(ops, op, dst, a, b),
            IrOp::MulHalf {
                dst,
                addend,
                a,
                b,
                a_high,
                b_high,
                long,
                set_flags,
            } => {
                if set_flags {
                    if !ensure_materialized_for_write(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_mul_half(ops, dst, addend, a, b, a_high, b_high, long, set_flags);
                if set_flags {
                    mode = FlagMode::Materialized;
                }
            }
            IrOp::MulWordHalf {
                dst,
                addend,
                a,
                b,
                b_high,
                set_flags,
            } => {
                if set_flags {
                    if !ensure_materialized_for_write(ops, &mut mode) {
                        return false;
                    }
                } else {
                    save_or_kill(ops, &mut mode, dead_after);
                }
                emit_mul_word_half(ops, dst, addend, a, b, b_high, set_flags);
                if set_flags {
                    mode = FlagMode::Materialized;
                }
            }
            IrOp::StoreMultiple {
                base,
                regs,
                add,
                pre_index,
                writeback,
            } => {
                save_or_kill(ops, &mut mode, dead_after);
                emit_store_multiple(ops, funcs, fault, base, regs, add, pre_index, writeback);
            }
            IrOp::Jump { target } => {
                save_or_kill(ops, &mut mode, dead_after);
                emit_set_pc(ops, target);
                match target {
                    // Only a static target can be linked; a dynamic one (`mov pc, lr`, `ldr pc, [...]`) has to go back to the driver
                    Value::Imm(pc) => {
                        emit_link_site(ops, epilogue, pc, thumb, block.start_pc, body, slots)
                    }
                    // A dynamic target is `BranchWritePC` (`MOV pc, Rm` before ARMv7), so the low bits
                    // go before it becomes PC — the interpreter does the same in
                    // `ArmState::branch_target`, and the two have to agree or a run differs from a
                    // replayed one. A static target comes from a branch encoding and is aligned
                    // already, which is why the linked path above needs no mask and pays nothing.
                    _ => {
                        if thumb {
                            dynasm!(ops ; and Rd(RESULT), !1 ; jmp => epilogue);
                        } else {
                            dynasm!(ops ; and Rd(RESULT), !3 ; jmp => epilogue);
                        }
                    }
                }
            }
            IrOp::JumpExchange { target } => {
                save_or_kill(ops, &mut mode, dead_after);
                emit_value(ops, target, RESULT, false);
                // PC = target & !1, T bit = target & 1
                dynasm!(ops
                    ; mov Rd(T2), Rd(RESULT)
                    ; and Rd(T2), 1
                    ; shl Rd(T2), 5
                    ; and Rd(CPSR), !0x20
                    ; or Rd(CPSR), Rd(T2)
                    ; and Rd(RESULT), !1
                );
                // Monomorphic inline cache: when the target matches the last one, jump straight into the **target's body**, skipping prologue and epilogue.
                // On a hit the register caches, STATE, CPSR and BUDGET are all still warm, exactly as at the landing point of a static link
                // (`body` exists precisely to be jumped into this way). A miss goes to the epilogue,
                // and the driver translates or compiles and then writes (pc, body) into the slot.
                let slot = exchange
                    .expect("a JumpExchange implies an exchange slot")
                    .cell_ptr() as u64;
                dynasm!(ops
                    ; mov Rq(T3), QWORD slot as i64
                    ; cmp Rd(RESULT), DWORD [Rq(T3)]
                    ; jne => epilogue
                    ; cmp Rd(BUDGET), 0
                    ; jle => epilogue
                    ; jmp QWORD [Rq(T3) + 8]
                );
            }
            IrOp::BranchConditional { cond, target } => {
                let taken = ops.new_dynamic_label();
                // The self-loop cold path needs sub_like to build the descriptor (the taken rebuild is outside the match)
                let mut x86_sub_like: Option<bool> = None;
                match mode {
                    FlagMode::X86 { sub_like } => {
                        if capture_condition_x86(ops, cond, sub_like) {
                            // Self-loop hot-path optimization: when the branch target is this block's start (the subs/bne shape)
                            // and the body rebuilds the flags (Discards), the taken fast path writes no descriptor and
                            // touches no memory, going straight back to the body; the descriptor store moves to the two cold exits:
                            //   1. fallthrough (leaving this block; the next one may read the cache)
                            //   2. budget exhaustion on the taken path (back to the driver through the epilogue, which wants it materialized)
                            // A hot iteration saves about ten instructions and two memory writes.
                            // The store has to happen before set_pc, which overwrites RESULT.
                            let self_loop = matches!(target, Value::Imm(pc) if pc == block.start_pc)
                                && block.entry_flags == EntryFlags::Discards;
                            if self_loop {
                                // Self-loop: both `test` and `cmp` destroy the x86 flags (TEST clears
                                // CF and OF), so C and V have to be copied into
                                // the two bytes T0 and T1 beforehand (setcc leaves the flags alone). The hot path therefore costs
                                // only two setccs instead of a whole descriptor.
                                dynasm!(ops ; setc Rb(T0) ; seto Rb(T1));
                            } else {
                                ensure_saved(ops, &mut mode);
                            }
                            dynasm!(ops
                                ; test Rb(T3), Rb(T3)
                                ; jnz => taken
                            );
                            if self_loop {
                                // Sequential exit: the x86 flags have been overwritten by the test, so only the captured
                                // C and V can rebuild the descriptor.
                                emit_descriptor_captured(ops, sub_like);
                                mode = FlagMode::Descriptor;
                                x86_sub_like = Some(sub_like);
                            }
                        } else {
                            if !ensure_materialized(ops, &mut mode) {
                                return false;
                            }
                            jump_if(ops, cond, taken);
                        }
                    }
                    FlagMode::Descriptor => jump_if_lazy(ops, cond, taken),
                    FlagMode::Materialized => jump_if(ops, cond, taken),
                    // The branch reads the flags: Dead (nobody reads them) and Captured (they are gone) both abandon compilation
                    FlagMode::Dead | FlagMode::Captured { .. } => return false,
                }
                dynasm!(ops ; jmp => fallthrough);
                dynasm!(ops ; => taken);
                // The self-loop taken path: test the budget first, and with budget left go back to the body for free (no store,
                // no PC write, since the body recomputes everything); only on exhaustion does it store, set_pc and return.
                // Non-self-loops take the original path.
                if let Some(sub_like) = x86_sub_like {
                    dynasm!(ops
                        ; cmp Rd(BUDGET), 0
                        ; jg => body
                    );
                    // Cold path: the budget is exhausted, so build the descriptor before returning to the driver (RESULT has not yet been
                    // overwritten by set_pc). ensure_saved cannot be used here: at compile time the mode has already moved to
                    // Descriptor in the fallthrough arm and would be a no-op; neither can emit_descriptor_flags be used,
                    // because the cmp above has already replaced CF and OF with the budget
                    // flags (this once wrote C as the inverse borrow flag; see tests/selfloop_review).
                    // (when a failed capture leaves the Materialized mode, CPSR is already correct, so the cold path
                    //   neither needs nor should write a descriptor.)
                    emit_descriptor_captured(ops, sub_like);
                    emit_set_pc(ops, target);
                    dynasm!(ops ; jmp => epilogue);
                } else {
                    emit_set_pc(ops, target);
                    match target {
                        Value::Imm(pc) => {
                            emit_link_site(ops, epilogue, pc, thumb, block.start_pc, body, slots)
                        }
                        _ => dynasm!(ops ; jmp => epilogue),
                    }
                }
            }
            // SVC and BKPT: record "pending exception + imm" in the state, set RESULT to this instruction's address
            // and go straight to the epilogue. The host resumes from RESULT once it has served the call.
            // Deferred flags become a descriptor first (if they are in x86) — an SVC touches no memory, so there is no ensure_saved
            // to fall back on, unlike the fault exit, which is safe by construction.
            IrOp::Swi { imm } | IrOp::Breakpoint { imm } => {
                ensure_saved(ops, &mut mode);
                let marker = (1u32 << 31)
                    | (u32::from(matches!(op, IrOp::Breakpoint { .. })) << 30)
                    | (imm & 0x00FF_FFFF);
                dynasm!(ops
                    ; mov DWORD [Rq(STATE) + EXC_OFF], marker as i32
                    ; mov Rd(RESULT), block.terminal_pc as i32
                );
                dynasm!(ops ; jmp => epilogue);
            } // The match covers every `IrOp` variant (the same source as is_supported);
              // when a variant is added, the uncovered-arm error here is the emitter's forced answer.
        }

        if trace {
            let bytes = ops.offset().0 - offset_before;
            let text = format!("{op:?}");
            let text: String = text.chars().take(72).collect();
            eprintln!("      [{index:2}] {bytes:4} bytes  {text}");
        }
        index += 1;
    }

    // A guard may extend to the block tail, in which case the merge point is the exit and the representation is restored as recorded
    while let Some(merge) = pending.pop() {
        dynasm!(ops ; => merge.label);
        mode = merge.mode;
    }

    // Sequential exit: if the flags are still in x86, turn them into a descriptor for the next block or a materialization
    if let FlagMode::X86 { sub_like } = mode {
        emit_descriptor_flags(ops, sub_like);
    }
    true
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

/// How the second operand affects the C flag.
#[derive(Clone, Copy)]
enum CarryOut {
    /// Does not define C (it keeps its value).
    Preserved,
    /// C is a compile-time constant (the carry of an immediate's rotation).
    Const(bool),
    /// C lives in the given register, already holding `0` or `Cpsr::C`.
    Reg(u8),
}

fn emit_value(ops: &mut Asm, value: Value, dst: u8, need_carry: bool) -> CarryOut {
    match value {
        Value::Imm(value) => {
            dynasm!(ops ; mov Rd(dst), value as i32);
            CarryOut::Preserved
        }
        Value::ImmC { value, carry } => {
            dynasm!(ops ; mov Rd(dst), value as i32);
            CarryOut::Const(carry)
        }
        Value::Reg(index) => {
            load_guest(ops, dst, index);
            CarryOut::Preserved
        }
        Value::Shifted {
            reg,
            shift,
            by: Shifter::Imm(amount),
        } => {
            load_guest(ops, dst, reg);

            if shift == ShiftType::Rrx {
                // RRX: the top bit comes from the current C and the carry out is the old value's bit 0
                if need_carry {
                    dynasm!(ops
                        ; mov Rd(T3), Rd(dst)
                        ; and Rd(T3), 1
                        ; shl Rd(T3), 29
                    );
                }
                dynasm!(ops
                    ; bt Rd(CPSR), 29
                    ; rcr Rd(dst), 1
                );
                return if need_carry {
                    CarryOut::Reg(T3)
                } else {
                    CarryOut::Preserved
                };
            }

            if amount == 0 {
                // Matching the interpreter: LSL #0 shifts nothing and C keeps its value
                return CarryOut::Preserved;
            }
            if need_carry {
                let bit = match shift {
                    ShiftType::Lsl => 32 - amount,
                    _ => amount - 1,
                };
                dynasm!(ops
                    ; mov Rd(T3), Rd(dst)
                    ; shr Rd(T3), bit as i8
                    ; and Rd(T3), 1
                    ; shl Rd(T3), 29
                );
            }
            if amount == 32 {
                // x86 truncates a shift count to five bits (32 becomes 0), so LSR and ASR by 32 must be expanded separately
                match shift {
                    ShiftType::Lsl | ShiftType::Lsr => dynasm!(ops ; xor Rd(dst), Rd(dst)),
                    ShiftType::Asr => dynasm!(ops ; sar Rd(dst), 31),
                    // ROR #32 is the identity; Rrx returned above
                    ShiftType::Ror | ShiftType::Rrx => {}
                }
            } else {
                match shift {
                    ShiftType::Lsl => dynasm!(ops ; shl Rd(dst), amount as i8),
                    ShiftType::Lsr => dynasm!(ops ; shr Rd(dst), amount as i8),
                    ShiftType::Asr => dynasm!(ops ; sar Rd(dst), amount as i8),
                    ShiftType::Ror => dynasm!(ops ; ror Rd(dst), amount as i8),
                    ShiftType::Rrx => unreachable!(),
                }
            }
            if need_carry {
                CarryOut::Reg(T3)
            } else {
                CarryOut::Preserved
            }
        }
        // A register-specified shift amount: x86's variable shifts need CL, so the amount goes into T2
        Value::Shifted {
            reg,
            shift,
            by: Shifter::Reg(rs),
        } => {
            load_guest(ops, dst, reg);
            load_guest(ops, T2, rs);
            // ARM uses only the low eight bits of the shift amount
            dynasm!(ops ; and Rd(T2), 0xFF);
            emit_shift_reg(ops, dst, shift, T2, need_carry)
        }
    }
}

/// The shifter for a register-specified amount (`lsl r0, r1, r2`). `amount` already holds the low eight bits.
///
/// Two places where x86 does not match, which have to be expanded by hand:
///
/// - x86 truncates a variable shift count to **five bits**, while ARM's 32 and >32 have different meanings;
/// - with a shift amount of 0 ARM **leaves C alone**, and the value is unchanged too.
///
/// The carry comes from `shld`/`shrd`, leaving the bits shifted out in a zeroed register,
/// which is one dependency chain shorter than computing `32 - amount` and shifting again.
fn emit_shift_reg(
    ops: &mut Asm,
    dst: u8,
    shift: ShiftType,
    amount: u8,
    need_carry: bool,
) -> CarryOut {
    // Variable shifts must use CL; the caller puts the amount into T2 (= RCX)
    debug_assert_eq!(amount, T2, "variable shifts support CL only");

    let zero = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();

    dynasm!(ops ; test Rd(amount), Rd(amount) ; jz => zero);

    match shift {
        // x86 reduces ROR's count modulo 32 automatically, which is exactly ARM's meaning (32 included: the value is unchanged)
        ShiftType::Ror => {
            dynasm!(ops ; ror Rd(dst), cl);
            if need_carry {
                dynasm!(ops ; mov Rd(T3), Rd(dst) ; shr Rd(T3), 31 ; shl Rd(T3), 29);
            }
            // The "shift amount is zero" block below must be skipped, or the carry would be overwritten with the old C
            dynasm!(ops ; jmp => done);
        }
        ShiftType::Lsl | ShiftType::Lsr | ShiftType::Asr => {
            let small = ops.new_dynamic_label();
            dynasm!(ops ; cmp Rd(amount), 32 ; jb => small);

            if need_carry {
                match shift {
                    // ASR's carry is the sign bit for anything >= 32
                    ShiftType::Asr => dynasm!(ops
                        ; mov Rd(T3), Rd(dst)
                        ; shr Rd(T3), 31
                        ; shl Rd(T3), 29
                    ),
                    // LSL and LSR: only **exactly** 32 produces a carry (bit 0 or bit 31); beyond that it is always 0
                    _ => {
                        let not_exact = ops.new_dynamic_label();
                        let after = ops.new_dynamic_label();
                        dynasm!(ops ; cmp Rd(amount), 32 ; jne => not_exact);
                        match shift {
                            ShiftType::Lsl => {
                                dynasm!(ops ; mov Rd(T3), Rd(dst) ; and Rd(T3), 1)
                            }
                            _ => dynasm!(ops ; mov Rd(T3), Rd(dst) ; shr Rd(T3), 31),
                        }
                        dynasm!(ops
                            ; shl Rd(T3), 29
                            ; jmp => after
                            ; => not_exact
                            ; xor Rd(T3), Rd(T3)
                            ; => after
                        );
                    }
                }
            }

            match shift {
                ShiftType::Asr => dynasm!(ops ; sar Rd(dst), 31),
                _ => dynasm!(ops ; xor Rd(dst), Rd(dst)),
            }
            dynasm!(ops ; jmp => done ; => small);

            if need_carry {
                match shift {
                    // shld leaves the bits shifted out in T3's low end, and bit 0 is the carry
                    ShiftType::Lsl => dynasm!(ops
                        ; xor Rd(T3), Rd(T3)
                        ; shld Rd(T3), Rd(dst), cl
                        ; and Rd(T3), 1
                        ; shl Rd(T3), 29
                    ),
                    // shrd is the mirror image: the carry lands in bit 31
                    _ => dynasm!(ops
                        ; xor Rd(T3), Rd(T3)
                        ; shrd Rd(T3), Rd(dst), cl
                        ; shr Rd(T3), 31
                        ; shl Rd(T3), 29
                    ),
                }
            }
            match shift {
                ShiftType::Lsl => dynasm!(ops ; shl Rd(dst), cl),
                ShiftType::Lsr => dynasm!(ops ; shr Rd(dst), cl),
                ShiftType::Asr => dynasm!(ops ; sar Rd(dst), cl),
                ShiftType::Ror | ShiftType::Rrx => unreachable!(),
            }
            dynasm!(ops ; jmp => done);
        }
        // The decoder never produces a register-shifted RRX
        ShiftType::Rrx => unreachable!(),
    }

    dynasm!(ops ; => zero);
    if need_carry {
        // A shift amount of zero: the value is unchanged and C keeps its value
        dynasm!(ops ; mov Rd(T3), Rd(CPSR) ; and Rd(T3), 0x2000_0000i32);
    }
    dynasm!(ops ; => done);

    if need_carry {
        CarryOut::Reg(T3)
    } else {
        CarryOut::Preserved
    }
}

/// The guest registers pinned permanently: `(guest register, host cache register)`.
///
/// A cache register has to be callee-saved (a block may call memory callbacks at any point), and x86-64 has few spare
/// callee-saved registers: on Windows only `RSI/RDI/R13` are left (arguments arrive in RCX/RDX),
/// and under SysV `RDI`/`RSI` are caller-saved argument registers, leaving only `R13`.
/// So Windows pins r0–r2 and SysV pins r0 alone.
#[cfg(target_os = "windows")]
const CACHE_REGS: [(u8, u8); 3] = [(0, 6), (1, 7), (2, 13)];
#[cfg(not(target_os = "windows"))]
const CACHE_REGS: [(u8, u8); 1] = [(0, 13)];

fn cache_of(index: u8) -> Option<u8> {
    CACHE_REGS
        .iter()
        .find(|(guest, _)| *guest == index)
        .map(|(_, host)| *host)
}

/// Reads a guest register: a pinned one comes from its host register, the rest from the state in memory.
fn load_guest(ops: &mut Asm, dst: u8, index: u8) {
    match cache_of(index) {
        Some(host) if host != dst => dynasm!(ops ; mov Rd(dst), Rd(host)),
        Some(_) => {}
        None => dynasm!(ops ; mov Rd(dst), [Rq(STATE) + reg_off(index)]),
    }
}

/// Writes a guest register: a pinned one goes only to its host register, and the memory copy is **stale for now**,
/// to be stored back by the epilogue (callbacks do not read `state.r`, and linking requires the mapping to agree; see `compile`).
fn store_guest(ops: &mut Asm, index: u8, src: u8) {
    match cache_of(index) {
        Some(host) if host != src => dynasm!(ops ; mov Rd(host), Rd(src)),
        Some(_) => {}
        None => dynasm!(ops ; mov [Rq(STATE) + reg_off(index)], Rd(src)),
    }
}

/// Writes every pinned register back to memory. Afterwards `[STATE + reg_off(..)]` can be used directly
/// (which one-off code such as the multiplies and saturating arithmetic uses instead of routing point by point).
fn spill_all_cached(ops: &mut Asm) {
    for (guest, host) in CACHE_REGS {
        dynasm!(ops ; mov [Rq(STATE) + reg_off(guest)], Rd(host));
    }
}

/// Reloads every pinned register from memory. Required after writing memory directly, or the cache holds stale values.
fn reload_all_cached(ops: &mut Asm) {
    for (guest, host) in CACHE_REGS {
        dynasm!(ops ; mov Rd(host), [Rq(STATE) + reg_off(guest)]);
    }
}

/// Computes the jump target into `EAX` as the block's return value (without writing `state.r[15]`; see [`JitFn`]).
fn emit_set_pc(ops: &mut Asm, target: Value) {
    emit_value(ops, target, RESULT, false);
}

// ---------------------------------------------------------------------------
// The operation kinds
// ---------------------------------------------------------------------------

/// The parameters are the IR operands plus two flag policies (`lazy` for deferred flags, `keep_flags` to leave the x86 flags alone);
/// a struct would only obscure how they line up.
#[allow(clippy::too_many_arguments)]
fn emit_alu(
    ops: &mut Asm,
    op: AluOp,
    dst: Option<u8>,
    a: Value,
    b: Option<Value>,
    set_flags: bool,
    lazy: bool,
    keep_flags: bool,
) {
    let logical_flags = op.is_logical() && set_flags;

    // MOV and MVN take their shifter carry from the first operand, everything else from the second
    let carry_from_a = logical_flags && op.is_move();
    let carry_from_b = logical_flags && !op.is_move();

    // The first operand is computed straight into RESULT, saving a `mov RESULT, VAL_A`.
    // Rsb, Rsc and Bic are the exception: they put the second operand into RESULT and compute backwards, leaving the first in VAL_A.
    let a_to_result = !matches!(op, AluOp::Rsb | AluOp::Rsc | AluOp::Bic);
    let carry_a = emit_value(
        ops,
        a,
        if a_to_result { RESULT } else { VAL_A },
        carry_from_a,
    );

    // The second operand is used as an x86 operand directly wherever possible (a guest register is already the memory operand `[state+off]`),
    // saving the instruction that would move it into VAL_B first. The Reg and Imm forms produce no shifter carry anyway.
    let (b_operand, carry_b) = match b {
        Some(Value::Reg(index)) => match cache_of(index) {
            // A pinned register becomes a register operand directly, saving one L1 access
            Some(host) => (Some(Operand::Reg(host)), CarryOut::Preserved),
            None => (Some(Operand::Mem(reg_off(index))), CarryOut::Preserved),
        },
        Some(Value::Imm(value)) => (Some(Operand::Imm(value as i32)), CarryOut::Preserved),
        Some(value) => (
            Some(Operand::Reg(VAL_B)),
            emit_value(ops, value, VAL_B, carry_from_b),
        ),
        None => (None, CarryOut::Preserved),
    };
    let shifter_carry = if op.is_move() { carry_a } else { carry_b };

    let invert_carry = matches!(
        op,
        AluOp::Sub | AluOp::Cmp | AluOp::Sbc | AluOp::Rsb | AluOp::Rsc
    );

    match op {
        AluOp::And | AluOp::Tst => emit_binop(ops, Mnemonic::And, b_operand.unwrap()),
        AluOp::Eor | AluOp::Teq => emit_binop(ops, Mnemonic::Xor, b_operand.unwrap()),
        AluOp::Orr => emit_binop(ops, Mnemonic::Or, b_operand.unwrap()),
        AluOp::Bic => {
            // a is in VAL_A; b goes into RESULT, is inverted, then and-ed back
            emit_load_operand(ops, b_operand.unwrap(), RESULT);
            dynasm!(ops ; not Rd(RESULT) ; and Rd(RESULT), Rd(VAL_A));
        }
        // The first operand is already in RESULT, so MOV needs no further instruction
        AluOp::Mov => {}
        AluOp::Mvn => dynasm!(ops ; not Rd(RESULT)),
        AluOp::Add | AluOp::Cmn => {
            if keep_flags {
                emit_add_keep_flags(ops, b_operand.unwrap());
            } else {
                emit_binop(ops, Mnemonic::Add, b_operand.unwrap());
            }
        }
        AluOp::Adc => {
            dynasm!(ops ; bt Rd(CPSR), 29);
            emit_binop(ops, Mnemonic::Adc, b_operand.unwrap());
        }
        AluOp::Sub | AluOp::Cmp => {
            if keep_flags {
                emit_sub_keep_flags(ops, b_operand.unwrap());
            } else {
                emit_binop(ops, Mnemonic::Sub, b_operand.unwrap());
            }
        }
        // ARM's C = 1 means "no borrow" while x86's CF = 1 means "borrow", so sbb's carry input has to be inverted:
        // ARM computes a - b - (1 - C) and x86 `sbb a, b` computes a - b - CF, so CF must be !C.
        AluOp::Sbc => {
            dynasm!(ops ; bt Rd(CPSR), 29 ; cmc);
            emit_binop(ops, Mnemonic::Sbb, b_operand.unwrap());
        }
        AluOp::Rsb => {
            emit_load_operand(ops, b_operand.unwrap(), RESULT);
            dynasm!(ops ; sub Rd(RESULT), Rd(VAL_A));
        }
        AluOp::Rsc => {
            emit_load_operand(ops, b_operand.unwrap(), RESULT);
            dynasm!(ops
                ; bt Rd(CPSR), 29
                ; cmc
                ; sbb Rd(RESULT), Rd(VAL_A)
            );
        }
    }

    if set_flags {
        if lazy {
            // The flags stay in the x86 flags, and emit_ops turns them into a descriptor when needed
        } else if op.is_logical() {
            emit_flags_logic(ops, shifter_carry);
        } else {
            emit_flags_arith(ops, invert_carry);
        }
    }

    if let Some(dst) = dst
        && !op.is_test()
    {
        store_guest(ops, dst, RESULT);
    }
}

/// The form the second operand takes in an x86 instruction.
///
/// A guest register serves as a **memory operand** directly (`add eax, [rbx+off]`), where every instruction used to have to
/// "move it into a temporary first", costing an instruction and register pressure.
#[derive(Clone, Copy)]
enum Operand {
    /// `[state + offset]`: the guest register lives in the state struct
    Mem(i32),
    Imm(i32),
    /// Already in a host register (a shifted operand, ImmC, or anything else that has to be computed first)
    Reg(u8),
}

#[derive(Clone, Copy)]
enum Mnemonic {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Adc,
    Sbb,
}

/// Emits `<mnemonic> RESULT, <b>`, with the first operand in `RESULT`.
fn emit_binop(ops: &mut Asm, mnemonic: Mnemonic, b: Operand) {
    match (mnemonic, b) {
        (Mnemonic::Add, Operand::Mem(offset)) => {
            dynasm!(ops ; add Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Add, Operand::Imm(imm)) => dynasm!(ops ; add Rd(RESULT), imm),
        (Mnemonic::Add, Operand::Reg(reg)) => dynasm!(ops ; add Rd(RESULT), Rd(reg)),
        (Mnemonic::Sub, Operand::Mem(offset)) => {
            dynasm!(ops ; sub Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Sub, Operand::Imm(imm)) => dynasm!(ops ; sub Rd(RESULT), imm),
        (Mnemonic::Sub, Operand::Reg(reg)) => dynasm!(ops ; sub Rd(RESULT), Rd(reg)),
        (Mnemonic::And, Operand::Mem(offset)) => {
            dynasm!(ops ; and Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::And, Operand::Imm(imm)) => dynasm!(ops ; and Rd(RESULT), imm),
        (Mnemonic::And, Operand::Reg(reg)) => dynasm!(ops ; and Rd(RESULT), Rd(reg)),
        (Mnemonic::Or, Operand::Mem(offset)) => {
            dynasm!(ops ; or Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Or, Operand::Imm(imm)) => dynasm!(ops ; or Rd(RESULT), imm),
        (Mnemonic::Or, Operand::Reg(reg)) => dynasm!(ops ; or Rd(RESULT), Rd(reg)),
        (Mnemonic::Xor, Operand::Mem(offset)) => {
            dynasm!(ops ; xor Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Xor, Operand::Imm(imm)) => dynasm!(ops ; xor Rd(RESULT), imm),
        (Mnemonic::Xor, Operand::Reg(reg)) => dynasm!(ops ; xor Rd(RESULT), Rd(reg)),
        (Mnemonic::Adc, Operand::Mem(offset)) => {
            dynasm!(ops ; adc Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Adc, Operand::Imm(imm)) => dynasm!(ops ; adc Rd(RESULT), imm),
        (Mnemonic::Adc, Operand::Reg(reg)) => dynasm!(ops ; adc Rd(RESULT), Rd(reg)),
        (Mnemonic::Sbb, Operand::Mem(offset)) => {
            dynasm!(ops ; sbb Rd(RESULT), [Rq(STATE) + offset])
        }
        (Mnemonic::Sbb, Operand::Imm(imm)) => dynasm!(ops ; sbb Rd(RESULT), imm),
        (Mnemonic::Sbb, Operand::Reg(reg)) => dynasm!(ops ; sbb Rd(RESULT), Rd(reg)),
    }
}

/// Moves the second operand into `dst` (Rsb/Rsc/Bic and the LEA paths need it as a register).
fn emit_load_operand(ops: &mut Asm, b: Operand, dst: u8) {
    match b {
        Operand::Mem(offset) => dynasm!(ops ; mov Rd(dst), [Rq(STATE) + offset]),
        Operand::Imm(imm) => dynasm!(ops ; mov Rd(dst), imm),
        Operand::Reg(reg) => {
            if reg != dst {
                dynasm!(ops ; mov Rd(dst), Rd(reg));
            }
        }
    }
}

/// `a + b` **without touching the x86 flags** (LEA writes none).
///
/// A conditional-execution block's `cmp; addcs; orrcs` relies on it: the flags are needed by the next guard
/// and must not be destroyed by an addition in between that does not care about them.
fn emit_add_keep_flags(ops: &mut Asm, b: Operand) {
    match b {
        Operand::Mem(offset) => dynasm!(ops
            ; mov Rd(VAL_B), [Rq(STATE) + offset]
            ; lea Rd(RESULT), [Rq(RESULT) + Rq(VAL_B)]
        ),
        Operand::Imm(imm) => dynasm!(ops ; lea Rd(RESULT), [Rq(RESULT) + imm]),
        Operand::Reg(reg) => dynasm!(ops ; lea Rd(RESULT), [Rq(RESULT) + Rq(reg)]),
    }
}

/// `a - b` **without touching the x86 flags**: `a + ~b + 1`, where neither `not` nor `lea` writes them.
fn emit_sub_keep_flags(ops: &mut Asm, b: Operand) {
    match b {
        // An immediate becomes a negative displacement directly: lea's disp may be negative
        Operand::Imm(imm) => dynasm!(ops ; lea Rd(RESULT), [Rq(RESULT) - imm]),
        Operand::Mem(offset) => dynasm!(ops
            ; mov Rd(VAL_B), [Rq(STATE) + offset]
            ; not Rd(VAL_B)
            ; lea Rd(RESULT), [Rq(RESULT) + Rq(VAL_B) + 1]
        ),
        Operand::Reg(reg) => dynasm!(ops
            ; not Rd(reg)
            ; lea Rd(RESULT), [Rq(RESULT) + Rq(reg) + 1]
        ),
    }
}

/// Whether the operand uses RRX, so its result depends on the current C.
fn value_reads_carry(value: Value) -> bool {
    matches!(
        value,
        Value::Shifted {
            shift: ShiftType::Rrx,
            ..
        }
    )
}

fn address_reads_carry(address: &Address) -> bool {
    match address {
        Address::Absolute(_) => false,
        Address::Register { offset, .. } => value_reads_carry(*offset),
    }
}

/// Whether the Alu reads C (ADC/SBC/RSC, or an RRX operand).
fn alu_reads_carry(op: AluOp, a: Value, b: Option<Value>) -> bool {
    matches!(op, AluOp::Adc | AluOp::Sbc | AluOp::Rsc)
        || value_reads_carry(a)
        || b.is_some_and(value_reads_carry)
}

/// An arithmetic instruction that writes all of NZCV and reads no C may defer its flags in the JIT.
fn alu_can_be_lazy(op: AluOp, a: Value, b: Option<Value>) -> bool {
    matches!(
        op,
        AluOp::Add | AluOp::Sub | AluOp::Rsb | AluOp::Cmp | AluOp::Cmn
    ) && !alu_reads_carry(op, a, b)
}

fn value_clobbers_x86(value: Value) -> bool {
    match value {
        Value::Imm(_) | Value::Reg(_) | Value::ImmC { .. } => false,
        Value::Shifted {
            shift: ShiftType::Lsl,
            by: Shifter::Imm(0),
            ..
        } => false,
        Value::Shifted { .. } => true,
    }
}

fn alu_clobbers_x86(op: AluOp, a: Value, b: Option<Value>) -> bool {
    if op.is_move() {
        value_clobbers_x86(a)
    } else {
        // Every other data-processing operation uses one of and/or/xor/add/sub, all of which destroy the x86 flags
        let _ = b;
        true
    }
}

/// Stores the result in the deferred descriptor, so it can still be materialized after an instruction that destroys the x86 flags.
fn emit_lazy_result_store(ops: &mut Asm) {
    dynasm!(ops ; mov [Rq(STATE) + LAZY_RESULT_OFF], Rd(RESULT));
}

/// Completes the deferred descriptor from the x86 flags just produced (the result is already in state.lazy.result).
fn emit_descriptor_flags(ops: &mut Asm, sub_like: bool) {
    dynasm!(ops
        ; setc Rb(T0)
        ; seto Rb(T1)
        ; movzx Rd(T0), Rb(T0)
        ; movzx Rd(T1), Rb(T1)
        ; shl Rd(T1), 1
        ; or Rd(T0), Rd(T1)
    );
    if sub_like {
        dynasm!(ops ; xor Rd(T0), 1);
    }
    dynasm!(ops
        ; or Rd(T0), LazyFlags::DIRTY as i32
        ; mov [Rq(STATE) + LAZY_FLAGS_OFF], Rd(T0)
    );
}

/// Completes the deferred descriptor from the two 0/1 bytes **captured earlier** in T0 (CF) and T1 (OF).
///
/// Unlike [`emit_descriptor_flags`] this reads no current x86 flag: it serves the case where "the flags would be overwritten
/// by an instruction in between (`test`/`cmp`) but C and V have to survive until the cold exit".
fn emit_descriptor_captured(ops: &mut Asm, sub_like: bool) {
    dynasm!(ops
        ; movzx Rd(T0), Rb(T0)
        ; movzx Rd(T1), Rb(T1)
        ; shl Rd(T1), 1
        ; or Rd(T0), Rd(T1)
    );
    if sub_like {
        dynasm!(ops ; xor Rd(T0), 1);
    }
    dynasm!(ops
        ; or Rd(T0), LazyFlags::DIRTY as i32
        ; mov [Rq(STATE) + LAZY_FLAGS_OFF], Rd(T0)
    );
}

/// If deferred flags exist, write them back to CPSR and clear the descriptor (for a block entry, where the descriptor is in the state).
fn emit_materialize_state(ops: &mut Asm) {
    let done = ops.new_dynamic_label();
    dynasm!(ops
        ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
        ; test Rd(T0), LazyFlags::DIRTY as i32
        ; jz => done
        ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
        ; mov Rd(VAL_A), Rd(T0)
        ; and Rd(VAL_A), 0x8000_0000u32 as i32
        ; cmp Rd(T0), 1
        ; sbb Rd(VAL_B), Rd(VAL_B)
        ; and Rd(VAL_B), 0x4000_0000u32 as i32
        ; or Rd(VAL_A), Rd(VAL_B)
        ; mov Rd(T1), [Rq(STATE) + LAZY_FLAGS_OFF]
        ; mov Rd(T2), Rd(T1)
        ; and Rd(T1), 1
        ; shl Rd(T1), 29
        ; or Rd(VAL_A), Rd(T1)
        ; shr Rd(T2), 1
        ; and Rd(T2), 1
        ; shl Rd(T2), 28
        ; or Rd(VAL_A), Rd(T2)
        ; and Rd(CPSR), 0x0FFF_FFFFu32 as i32
        ; or Rd(CPSR), Rd(VAL_A)
        ; mov DWORD [Rq(STATE) + LAZY_FLAGS_OFF], 0
        ; => done
    );
}

/// Materializes the x86 flags (and the result kept in the state) into CPSR.
fn emit_materialize_x86(ops: &mut Asm, sub_like: bool) {
    dynasm!(ops ; mov Rd(RESULT), [Rq(STATE) + LAZY_RESULT_OFF]);
    emit_flags_arith(ops, sub_like);
    dynasm!(ops ; mov DWORD [Rq(STATE) + LAZY_FLAGS_OFF], 0);
}

fn emit_mul(ops: &mut Asm, dst: u8, a: u8, b: u8, accumulate: Option<u8>, set_flags: bool) {
    spill_all_cached(ops);
    dynasm!(ops
        ; mov Rd(RESULT), [Rq(STATE) + reg_off(a)]
        ; imul Rd(RESULT), [Rq(STATE) + reg_off(b)]
    );
    if let Some(acc) = accumulate {
        dynasm!(ops ; add Rd(RESULT), [Rq(STATE) + reg_off(acc)]);
    }
    store_guest(ops, dst, RESULT);

    if set_flags {
        // IMUL leaves SF and ZF undefined, so N and Z are computed here
        emit_nz_only(ops);
    }
}

/// Updates only N and Z, leaving C and V (the partial update the multiplies and halfword multiplies perform).
fn emit_nz_only(ops: &mut Asm) {
    dynasm!(ops
        ; mov Rd(VAL_A), Rd(RESULT)
        ; and Rd(VAL_A), 0x8000_0000u32 as i32
        ; cmp Rd(RESULT), 1
        ; sbb Rd(VAL_B), Rd(VAL_B)
        ; and Rd(VAL_B), 0x4000_0000u32 as i32
        ; or Rd(VAL_A), Rd(VAL_B)
        ; and Rd(CPSR), 0x3FFF_FFFFu32 as i32
        ; or Rd(CPSR), Rd(VAL_A)
    );
}

/// Takes a halfword and sign-extends it: `high` false takes the low 16 bits, true the top 16.
fn emit_signed_half(ops: &mut Asm, dst: u8, reg: u8, high: bool) {
    load_guest(ops, dst, reg);
    if high {
        dynasm!(ops ; sar Rd(dst), 16);
    } else {
        dynasm!(ops ; shl Rd(dst), 16 ; sar Rd(dst), 16);
    }
}

/// SMULxy, SMLAxy and SMLALxy.
#[allow(clippy::too_many_arguments)]
fn emit_mul_half(
    ops: &mut Asm,
    dst: u8,
    addend: Option<u8>,
    a: u8,
    b: u8,
    a_high: bool,
    b_high: bool,
    long: bool,
    set_flags: bool,
) {
    emit_signed_half(ops, T0, a, a_high);
    emit_signed_half(ops, T1, b, b_high);
    // The product of two signed 16-bit values fits in 30 bits, so a 32-bit imul cannot overflow
    dynasm!(ops ; imul Rd(T0), Rd(T1));

    spill_all_cached(ops);
    if long {
        // SMLALxy: RdHi:RdLo += the product (written to memory directly, with the cache reloaded at the exit)
        dynasm!(ops
            ; movsxd Rq(T0), Rd(T0)
            ; mov Rd(RESULT), [Rq(STATE) + reg_off(dst + 1)]
            ; shl Rq(RESULT), 32
            ; mov Rd(T1), [Rq(STATE) + reg_off(dst)]
            ; or Rq(RESULT), Rq(T1)
            ; add Rq(RESULT), Rq(T0)
            ; mov [Rq(STATE) + reg_off(dst)], Rd(RESULT)
            ; shr Rq(RESULT), 32
            ; mov [Rq(STATE) + reg_off(dst + 1)], Rd(RESULT)
        );
        reload_all_cached(ops);
        return;
    }

    dynasm!(ops ; mov Rd(RESULT), Rd(T0));
    if let Some(acc) = addend {
        dynasm!(ops
            ; add Rd(RESULT), [Rq(STATE) + reg_off(acc)]
            // SMLAxy: set Q when the accumulation overflows 32 bits. `seto` has to follow `add` immediately,
            // because the N/Z computation below (cmp/sbb) would overwrite OF. The product is only 30 bits wide, so only this addition can overflow.
            ; seto Rb(T1)
            ; movzx Rd(T1), Rb(T1)
            ; shl Rd(T1), 27
            ; or Rd(CPSR), Rd(T1)
        );
    }
    store_guest(ops, dst, RESULT);
    if set_flags {
        emit_nz_only(ops);
    }
}

/// SMULWy and SMLAWy: `(Rm × Rs.half) >> 16`, optionally adding Ra.
fn emit_mul_word_half(
    ops: &mut Asm,
    dst: u8,
    addend: Option<u8>,
    a: u8,
    b: u8,
    b_high: bool,
    set_flags: bool,
) {
    spill_all_cached(ops);
    dynasm!(ops ; movsxd Rq(RESULT), [Rq(STATE) + reg_off(a)]);
    emit_signed_half(ops, T1, b, b_high);
    dynasm!(ops
        ; movsxd Rq(T1), Rd(T1)
        ; imul Rq(RESULT), Rq(T1)
        ; sar Rq(RESULT), 16
    );
    if let Some(acc) = addend {
        dynasm!(ops
            ; add Rd(RESULT), [Rq(STATE) + reg_off(acc)]
            // SMLAWy likewise: set Q on accumulation overflow (seto has to follow add immediately)
            ; seto Rb(T1)
            ; movzx Rd(T1), Rb(T1)
            ; shl Rd(T1), 27
            ; or Rd(CPSR), Rd(T1)
        );
    }
    store_guest(ops, dst, RESULT);
    if set_flags {
        emit_nz_only(ops);
    }
}

/// Saturates the 64-bit value in `Rq(RESULT)` into the i32 range, setting `acc` to 1 whenever it saturates.
///
/// Equivalent to the interpreter: test the upper bound first, then the lower, and the boundary values themselves are not overflow.
fn emit_saturate(ops: &mut Asm, acc: u8) {
    let hi = ops.new_dynamic_label();
    let lo = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();
    dynasm!(ops
        ; cmp Rq(RESULT), 0x7fff_ffffi32
        ; jg => hi
        ; cmp Rq(RESULT), i32::MIN
        ; jl => lo
        ; jmp => done
        ; => hi
        ; mov Rd(RESULT), 0x7fff_ffff
        ; mov Rd(acc), 1
        ; jmp => done
        ; => lo
        ; mov Rq(RESULT), QWORD (-0x8000_0000i64)
        ; mov Rd(acc), 1
        ; => done
    );
}

/// QADD, QSUB, QDADD and QDSUB: `Rd = sat(Rm ± Rn)`, with `QD*` doubling Rn first and saturating at each step.
///
/// The semantics follow the interpreter: **first** saturate `2 × Rn` into i32, then add or subtract Rm and saturate again;
/// any saturation sets CPSR.Q (Q is only ever set, never cleared).
fn emit_saturating(ops: &mut Asm, op: SatOp, dst: u8, a: u8, b: u8) {
    spill_all_cached(ops);
    let doubled = matches!(op, SatOp::DoubleAdd | SatOp::DoubleSub);
    let subtract = matches!(op, SatOp::Sub | SatOp::DoubleSub);

    dynasm!(ops
        ; movsxd Rq(RESULT), [Rq(STATE) + reg_off(b)]
        ; movsxd Rq(T2), [Rq(STATE) + reg_off(a)]
        ; xor Rd(T1), Rd(T1)
    );
    if doubled {
        dynasm!(ops
            ; shl Rq(T2), 1
            ; mov Rq(T3), Rq(RESULT)
            ; mov Rq(RESULT), Rq(T2)
        );
        emit_saturate(ops, T1);
        dynasm!(ops
            ; mov Rq(T2), Rq(RESULT)
            ; mov Rq(RESULT), Rq(T3)
        );
    }
    if subtract {
        dynasm!(ops ; sub Rq(RESULT), Rq(T2));
    } else {
        dynasm!(ops ; add Rq(RESULT), Rq(T2));
    }
    emit_saturate(ops, T1);
    store_guest(ops, dst, RESULT);

    let done = ops.new_dynamic_label();
    dynasm!(ops
        ; test Rd(T1), Rd(T1)
        ; jz => done
        ; or Rd(CPSR), Cpsr::Q as i32
        ; => done
    );
}

/// SWP and SWPB: read the old value, write the new one, and put the old one in Rd.
///
/// The order follows the interpreter: `state.r[src]` is read **after** the read, so `Rd == Rm` behaves the same way.
fn emit_swap(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    dst: u8,
    src: u8,
    address: Address,
    byte: bool,
) {
    emit_address(ops, address, true);
    // The callback destroys EFF (volatile), so both the address and the old value cross over through the spill area
    dynasm!(ops ; mov [Rq(4) + SPILL_OFF], Rd(EFF));
    emit_read_value(
        ops,
        funcs,
        fault,
        if byte { MemSize::U8 } else { MemSize::U32 },
    );
    dynasm!(ops
        ; mov [Rq(4) + SPILL_OFF + 8], Rd(RESULT)
        ; mov Rd(EFF), [Rq(4) + SPILL_OFF]
    );
    load_guest(ops, T3, src);
    emit_write_value(
        ops,
        funcs,
        fault,
        if byte { MemSize::U8 } else { MemSize::U32 },
        T3,
    );
    dynasm!(ops ; mov Rd(RESULT), [Rq(4) + SPILL_OFF + 8]);
    store_guest(ops, dst, RESULT);
}

/// LDRD: two consecutive words into `rt`:`rt+1`.
fn emit_load_double(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    rt: u8,
    address: Address,
    pre_index: bool,
    writeback: bool,
) {
    let writeback_info = emit_address(ops, address, pre_index);
    if writeback && let Some((base, offset_addr)) = writeback_info {
        store_guest(ops, base, offset_addr);
    }
    dynasm!(ops ; mov [Rq(4) + SPILL_OFF], Rd(EFF));

    emit_read_value(ops, funcs, fault, MemSize::U32);
    store_guest(ops, rt, RESULT);

    dynasm!(ops
        ; mov Rd(EFF), [Rq(4) + SPILL_OFF]
        ; add Rd(EFF), 4
    );
    emit_read_value(ops, funcs, fault, MemSize::U32);
    store_guest(ops, rt + 1, RESULT);
}

/// STRD: writes the two consecutive words of `rt`:`rt+1`.
fn emit_store_double(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    rt: u8,
    address: Address,
    pre_index: bool,
    writeback: bool,
) {
    // Both values must be taken **before the writeback**: the base register may itself be in rt:rt+1
    // (`strd r0, [r1, #8]!` stores r0 and r1), the low word in T3 and the high word in the spill area —
    // `emit_address` uses T1 as a scratch register, and the callback destroys every volatile register.
    load_guest(ops, T3, rt);
    load_guest(ops, T1, rt + 1);
    dynasm!(ops ; mov [Rq(4) + SPILL_OFF + 8], Rd(T1));

    let writeback_info = emit_address(ops, address, pre_index);
    if writeback && let Some((base, offset_addr)) = writeback_info {
        store_guest(ops, base, offset_addr);
    }
    dynasm!(ops ; mov [Rq(4) + SPILL_OFF], Rd(EFF));

    emit_write_value(ops, funcs, fault, MemSize::U32, T3);

    dynasm!(ops
        ; mov Rd(T3), [Rq(4) + SPILL_OFF + 8]
        ; mov Rd(EFF), [Rq(4) + SPILL_OFF]
        ; add Rd(EFF), 4
    );
    emit_write_value(ops, funcs, fault, MemSize::U32, T3);
}

#[allow(clippy::too_many_arguments)]
fn emit_mul_long(
    ops: &mut Asm,
    lo: u8,
    hi: u8,
    a: u8,
    b: u8,
    signed: bool,
    accumulate: bool,
    set_flags: bool,
) {
    spill_all_cached(ops);
    if signed {
        dynasm!(ops
            ; movsxd Rq(RESULT), [Rq(STATE) + reg_off(a)]
            ; movsxd Rq(T2), [Rq(STATE) + reg_off(b)]
        );
    } else {
        dynasm!(ops
            ; mov Rd(RESULT), [Rq(STATE) + reg_off(a)]
            ; mov Rd(T2), [Rq(STATE) + reg_off(b)]
        );
    }
    dynasm!(ops ; imul Rq(RESULT), Rq(T2));

    if accumulate {
        dynasm!(ops
            ; mov Rq(T0), [Rq(STATE) + reg_off(hi)]
            ; shl Rq(T0), 32
            ; mov Rd(T1), [Rq(STATE) + reg_off(lo)]
            ; or Rq(T0), Rq(T1)
            ; add Rq(RESULT), Rq(T0)
        );
    }

    dynasm!(ops
        ; mov [Rq(STATE) + reg_off(lo)], Rd(RESULT)
        ; mov Rq(T0), Rq(RESULT)
        ; shr Rq(T0), 32
        ; mov [Rq(STATE) + reg_off(hi)], Rd(T0)
    );
    reload_all_cached(ops);

    if set_flags {
        // N = bit 63, Z = the whole 64-bit value being zero
        dynasm!(ops
            ; mov Rq(T0), Rq(RESULT)
            ; shr Rq(T0), 63
            ; shl Rd(T0), 31
            ; test Rq(RESULT), Rq(RESULT)
            ; sete Rb(T1)
            ; movzx Rd(T1), Rb(T1)
            ; shl Rd(T1), 30
            ; or Rd(T0), Rd(T1)
            ; and Rd(CPSR), 0x3FFF_FFFFu32 as i32
            ; or Rd(CPSR), Rd(T0)
        );
    }
}

fn emit_clz(ops: &mut Asm, dst: u8, src: u8) {
    let zero = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();
    load_guest(ops, RESULT, src);
    dynasm!(ops
        ; bsr Rd(T2), Rd(RESULT)
        ; jz => zero
        ; mov Rd(RESULT), 31
        ; sub Rd(RESULT), Rd(T2)
        ; jmp => done
        ; => zero
        ; mov Rd(RESULT), 32
        ; => done
    );
    store_guest(ops, dst, RESULT);
}

fn emit_write_status(ops: &mut Asm, src: Value, mask: u32, spsr: bool) {
    emit_value(ops, src, RESULT, false);
    dynasm!(ops ; and Rd(RESULT), mask as i32);
    if spsr {
        dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + SPSR_OFF]
            ; and Rd(T0), !mask as i32
            ; or Rd(T0), Rd(RESULT)
            ; mov [Rq(STATE) + SPSR_OFF], Rd(T0)
        );
    } else {
        dynasm!(ops
            ; and Rd(CPSR), !mask as i32
            ; or Rd(CPSR), Rd(RESULT)
        );
    }
}

// ---------------------------------------------------------------------------
// The flags
// ---------------------------------------------------------------------------

/// Updates N/Z/C/V after arithmetic. The result is in [`RESULT`] and the x86 flags were just set by `add`/`sub`.
///
/// `invert_carry` is for the subtraction family: x86's CF is a **borrow** and ARM's C is its inverse.
fn emit_flags_arith(ops: &mut Asm, invert_carry: bool) {
    dynasm!(ops
        ; setc Rb(T0)
        ; seto Rb(T1)
        ; movzx Rd(T0), Rb(T0)
        ; movzx Rd(T1), Rb(T1)
    );
    if invert_carry {
        dynasm!(ops ; xor Rd(T0), 1);
    }
    dynasm!(ops
        ; shl Rd(T0), 29
        ; shl Rd(T1), 28
        ; mov Rd(VAL_A), Rd(RESULT)
        ; and Rd(VAL_A), 0x8000_0000u32 as i32
        ; cmp Rd(RESULT), 1
        ; sbb Rd(VAL_B), Rd(VAL_B)
        ; and Rd(VAL_B), 0x4000_0000u32 as i32
        ; or Rd(VAL_A), Rd(VAL_B)
        ; or Rd(VAL_A), Rd(T0)
        ; or Rd(VAL_A), Rd(T1)
        ; and Rd(CPSR), 0x0FFF_FFFFu32 as i32
        ; or Rd(CPSR), Rd(VAL_A)
    );
}

/// Updates N and Z after a logical operation and handles C from `carry` (V is left alone).
fn emit_flags_logic(ops: &mut Asm, carry: CarryOut) {
    dynasm!(ops
        ; mov Rd(VAL_A), Rd(RESULT)
        ; and Rd(VAL_A), 0x8000_0000u32 as i32
        ; cmp Rd(RESULT), 1
        ; sbb Rd(VAL_B), Rd(VAL_B)
        ; and Rd(VAL_B), 0x4000_0000u32 as i32
        ; or Rd(VAL_A), Rd(VAL_B)
    );

    let mut mask: u32 = 0xC000_0000;
    match carry {
        CarryOut::Preserved => {}
        CarryOut::Const(true) => {
            dynasm!(ops ; or Rd(VAL_A), 0x2000_0000i32);
            mask |= 0x2000_0000;
        }
        CarryOut::Const(false) => {
            mask |= 0x2000_0000;
        }
        CarryOut::Reg(reg) => {
            dynasm!(ops ; or Rd(VAL_A), Rd(reg));
            mask |= 0x2000_0000;
        }
    }

    dynasm!(ops
        ; and Rd(CPSR), !(mask as i32)
        ; or Rd(CPSR), Rd(VAL_A)
    );
}

// ---------------------------------------------------------------------------
// Conditional jumps
// ---------------------------------------------------------------------------

/// Jumps to `label` when the condition holds.
///
/// ARM encodes a condition code as "drop the low bit, then invert", so `cond ^ 1` is always its complement.
fn jump_if(ops: &mut Asm, cond: u32, label: dynasmrt::DynamicLabel) {
    match cond {
        0x0 => jump_bit(ops, 30, true, label),  // EQ: Z
        0x1 => jump_bit(ops, 30, false, label), // NE
        0x2 => jump_bit(ops, 29, true, label),  // CS
        0x3 => jump_bit(ops, 29, false, label), // CC
        0x4 => jump_bit(ops, 31, true, label),  // MI
        0x5 => jump_bit(ops, 31, false, label), // PL
        0x6 => jump_bit(ops, 28, true, label),  // VS
        0x7 => jump_bit(ops, 28, false, label), // VC
        0x8 => {
            // HI: C && !Z
            let skip = ops.new_dynamic_label();
            dynasm!(ops
                ; bt Rd(CPSR), 30
                ; jc => skip
                ; bt Rd(CPSR), 29
                ; jc => label
                ; => skip
            );
        }
        0x9 => {
            // LS: !C || Z
            dynasm!(ops
                ; bt Rd(CPSR), 30
                ; jc => label
                ; bt Rd(CPSR), 29
                ; jnc => label
            );
        }
        0xA => emit_ge_lt(ops, true, label),  // GE: N == V
        0xB => emit_ge_lt(ops, false, label), // LT: N != V
        0xC => {
            // GT: !Z && N == V
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; bt Rd(CPSR), 30
                ; jc => done
            );
            emit_ge_lt(ops, true, label);
            dynasm!(ops ; => done);
        }
        0xD => {
            // LE: Z || N != V
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; bt Rd(CPSR), 30
                ; jc => label
            );
            emit_ge_lt(ops, false, label);
            dynasm!(ops ; => done);
        }
        _ => dynasm!(ops ; jmp => label),
    }
}

fn jump_bit(ops: &mut Asm, bit: u8, set: bool, label: dynasmrt::DynamicLabel) {
    dynasm!(ops ; bt Rd(CPSR), bit as i8);
    if set {
        dynasm!(ops ; jc => label);
    } else {
        dynasm!(ops ; jnc => label);
    }
}

/// Jumps when `N == V` (`same == true`) or when `N != V`.
fn emit_ge_lt(ops: &mut Asm, same: bool, label: dynasmrt::DynamicLabel) {
    let v_set = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();
    dynasm!(ops
        ; bt Rd(CPSR), 28
        ; jc => v_set
        // V == 0: N == 0 means N == V
        ; bt Rd(CPSR), 31
    );
    if same {
        dynasm!(ops ; jnc => label);
    } else {
        dynasm!(ops ; jc => label);
    }
    dynasm!(ops
        ; jmp => done
        ; => v_set
        // V == 1: N == 1 means N == V
        ; bt Rd(CPSR), 31
    );
    if same {
        dynasm!(ops ; jc => label);
    } else {
        dynasm!(ops ; jnc => label);
    }
    dynasm!(ops ; => done);
}

/// Tests an ARM condition against the x86 flags directly and jumps.
///
/// The key property is that `jcc` **does not modify** the flags, so a condition guard can sit between "the flags were just computed" and "the next read of them"
/// without having to pack them into a descriptor first. That is what lets conditional-execution blocks use deferred flags.
///
/// `sub_like` says whether x86's CF is a borrow: after a subtraction ARM's C is the opposite of CF.
/// `false` means this condition has no single matching `jcc` under the current carry direction, and the caller should materialize first.
fn jump_if_x86(ops: &mut Asm, cond: u32, sub_like: bool, label: dynasmrt::DynamicLabel) -> bool {
    match cond {
        0x0 => dynasm!(ops ; jz => label),              // EQ: ZF
        0x1 => dynasm!(ops ; jnz => label),             // NE
        0x2 if sub_like => dynasm!(ops ; jnc => label), // CS: C = !CF
        0x2 => dynasm!(ops ; jc => label),
        0x3 if sub_like => dynasm!(ops ; jc => label), // CC
        0x3 => dynasm!(ops ; jnc => label),
        0x4 => dynasm!(ops ; js => label),  // MI
        0x5 => dynasm!(ops ; jns => label), // PL
        0x6 => dynasm!(ops ; jo => label),  // VS
        0x7 => dynasm!(ops ; jno => label), // VC
        // HI = C && !Z. After a subtraction C = !CF, so HI = !CF && !ZF = ja.
        // After an addition C = CF, and HI = CF && !ZF has no single jcc, so return false and let the caller materialize.
        0x8 if sub_like => dynasm!(ops ; ja => label),
        // LS = !C || Z: after a subtraction !C = CF and after an addition !C = !CF — either way it lands on jbe
        0x9 if sub_like => dynasm!(ops ; jbe => label),
        0xA => dynasm!(ops ; jge => label), // GE: SF == OF
        0xB => dynasm!(ops ; jl => label),  // LT
        0xC => dynasm!(ops ; jg => label),  // GT: !ZF && SF == OF
        0xD => dynasm!(ops ; jle => label), // LE
        0xE | 0xF => dynasm!(ops ; jmp => label),
        _ => return false,
    }
    true
}

/// Tests the condition from the deferred descriptor and jumps.
fn jump_if_lazy(ops: &mut Asm, cond: u32, label: dynasmrt::DynamicLabel) {
    match cond {
        0x0 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
            ; test Rd(T0), Rd(T0)
            ; jz => label
        ),
        0x1 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
            ; test Rd(T0), Rd(T0)
            ; jnz => label
        ),
        0x2 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
            ; test Rd(T0), 1
            ; jnz => label
        ),
        0x3 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
            ; test Rd(T0), 1
            ; jz => label
        ),
        0x4 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
            ; test Rd(T0), 0x8000_0000u32 as i32
            ; jnz => label
        ),
        0x5 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
            ; test Rd(T0), 0x8000_0000u32 as i32
            ; jz => label
        ),
        0x6 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
            ; test Rd(T0), 2
            ; jnz => label
        ),
        0x7 => dynasm!(ops
            ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
            ; test Rd(T0), 2
            ; jz => label
        ),
        0x8 => {
            // HI: C && !Z
            let skip = ops.new_dynamic_label();
            dynasm!(ops
                ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
                ; test Rd(T0), Rd(T0)
                ; jz => skip
                ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
                ; test Rd(T0), 1
                ; jnz => label
                ; => skip
            );
        }
        0x9 => {
            // LS: !C || Z
            dynasm!(ops
                ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
                ; test Rd(T0), Rd(T0)
                ; jz => label
                ; mov Rd(T0), [Rq(STATE) + LAZY_FLAGS_OFF]
                ; test Rd(T0), 1
                ; jz => label
            );
        }
        0xA => emit_ge_lt_lazy(ops, true, label), // GE: N == V
        0xB => emit_ge_lt_lazy(ops, false, label), // LT: N != V
        0xC => {
            // GT: !Z && N == V
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
                ; test Rd(T0), Rd(T0)
                ; jz => done
            );
            emit_ge_lt_lazy(ops, true, label);
            dynasm!(ops ; => done);
        }
        0xD => {
            // LE: Z || N != V
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
                ; test Rd(T0), Rd(T0)
                ; jz => label
            );
            emit_ge_lt_lazy(ops, false, label);
            dynasm!(ops ; => done);
        }
        _ => dynasm!(ops ; jmp => label),
    }
}

/// `N == V` and `N != V` under the deferred descriptor.
fn emit_ge_lt_lazy(ops: &mut Asm, same: bool, label: dynasmrt::DynamicLabel) {
    dynasm!(ops
        ; mov Rd(T0), [Rq(STATE) + LAZY_RESULT_OFF]
        ; mov Rd(T1), [Rq(STATE) + LAZY_FLAGS_OFF]
        ; shl Rd(T1), 30
        ; xor Rd(T0), Rd(T1)
        ; test Rd(T0), 0x8000_0000u32 as i32
    );
    if same {
        dynasm!(ops ; jz => label);
    } else {
        dynasm!(ops ; jnz => label);
    }
}

/// Captures an ARM condition into T3's low byte without modifying the x86 flags.
/// `false` means the condition cannot be expressed with a single `setcc`.
fn capture_condition_x86(ops: &mut Asm, cond: u32, sub_like: bool) -> bool {
    match cond {
        0x0 => dynasm!(ops ; sete Rb(T3)),
        0x1 => dynasm!(ops ; setne Rb(T3)),
        0x2 => {
            if sub_like {
                dynasm!(ops ; setnc Rb(T3))
            } else {
                dynasm!(ops ; setc Rb(T3))
            }
        }
        0x3 => {
            if sub_like {
                dynasm!(ops ; setc Rb(T3))
            } else {
                dynasm!(ops ; setnc Rb(T3))
            }
        }
        0x4 => dynasm!(ops ; sets Rb(T3)),
        0x5 => dynasm!(ops ; setns Rb(T3)),
        0x6 => dynasm!(ops ; seto Rb(T3)),
        0x7 => dynasm!(ops ; setno Rb(T3)),
        0x8 => {
            if sub_like {
                dynasm!(ops ; seta Rb(T3))
            } else {
                return false;
            }
        }
        0x9 => {
            if sub_like {
                dynasm!(ops ; setbe Rb(T3))
            } else {
                return false;
            }
        }
        0xA => dynasm!(ops ; setge Rb(T3)),
        0xB => dynasm!(ops ; setl Rb(T3)),
        0xC => dynasm!(ops ; setg Rb(T3)),
        0xD => dynasm!(ops ; setle Rb(T3)),
        _ => {
            dynasm!(ops ; mov Rb(T3), 1);
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Memory access
// ---------------------------------------------------------------------------

/// The effective-address register. It carries the address into a callback; being volatile, it is unusable afterwards.
const EFF: u8 = R9;
/// The temporary for address computation.
///
/// It **is the same register as `EFF`**: the finished address already sits there, saving a move
/// (`emit_address` is always followed immediately by the access, so there is no need to distinguish "temporary" from "effective").
const ADDR_TMP: u8 = EFF;

fn size_index(size: MemSize) -> usize {
    match size {
        MemSize::U8 => 0,
        MemSize::U16 => 1,
        MemSize::U32 => 2,
    }
}

/// Computes the effective address into [`EFF`] and returns the writeback information `(base register, register holding offset_addr)`.
///
/// The caller has to perform the writeback **before the callback**: `offset_addr` lives in a volatile register
/// that the callback destroys. Semantically it is equivalent anyway — the effective address is already computed, so the writeback cannot affect the access.
fn emit_address(ops: &mut Asm, address: Address, pre_index: bool) -> Option<(u8, u8)> {
    match address {
        Address::Absolute(addr) => {
            dynasm!(ops ; mov Rd(EFF), addr as i32);
            None
        }
        Address::Register { base, offset, add } => {
            load_guest(ops, ADDR_TMP, base);
            // An immediate offset is the commonest form (`[r1, #4]`, `[sp]`): one add or lea does it directly,
            // saving the two instructions of "move into a temporary, then add".
            if let Value::Imm(value) = offset {
                let delta = if add { value as i32 } else { -(value as i32) };
                if pre_index {
                    if delta != 0 {
                        dynasm!(ops ; add Rd(ADDR_TMP), delta);
                    }
                    return Some((base, ADDR_TMP));
                }
                // Post-indexing: the access uses the base itself (already in EFF) and the writeback value is "base + offset"
                if delta == 0 {
                    return Some((base, ADDR_TMP));
                }
                dynasm!(ops ; lea Rd(T0), [Rq(ADDR_TMP) + delta]);
                return Some((base, T0));
            }
            emit_value(ops, offset, T0, false);
            if add {
                // Addition commutes, so accumulate directly
                dynasm!(ops ; add Rd(T0), Rd(ADDR_TMP));
            } else {
                // Subtraction does not commute: base - offset has to be computed while the base in ADDR_TMP is still needed
                dynasm!(ops
                    ; mov Rd(T1), Rd(ADDR_TMP)
                    ; sub Rd(T1), Rd(T0)
                    ; mov Rd(T0), Rd(T1)
                );
            }
            let effective = if pre_index { T0 } else { ADDR_TMP };
            if effective != EFF {
                dynasm!(ops ; mov Rd(EFF), Rd(effective));
            }
            Some((base, T0))
        }
    }
}

fn emit_call_read(ops: &mut Asm, func: usize, fault: dynasmrt::DynamicLabel) {
    #[cfg(windows)]
    dynasm!(ops
        ; mov Rq(RCX), Rq(MEM)
        ; mov Rd(RDX), Rd(EFF)
        ; mov Rq(RAX), QWORD (func as i64)
        ; call Rq(RAX)
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; mov Rq(7), Rq(MEM)
        ; mov Rd(6), Rd(EFF)
        ; mov Rq(RAX), QWORD (func as i64)
        ; call Rq(RAX)
    );

    // Check for a host-reported access fault as soon as the callback returns: one `cmp` and one `jne`,
    // present only on accesses that **go through a callback** (a windowed inline is mapped by contract and is not checked).
    dynasm!(ops
        ; cmp DWORD [Rq(STATE) + FAULT_OFF], 0
        ; jne => fault
    );
}

fn emit_call_write(ops: &mut Asm, func: usize, value: u8, fault: dynasmrt::DynamicLabel) {
    #[cfg(windows)]
    dynasm!(ops
        ; mov Rq(RCX), Rq(MEM)
        ; mov Rd(RDX), Rd(EFF)
        ; mov Rd(R8), Rd(value)
        ; mov Rq(RAX), QWORD (func as i64)
        ; call Rq(RAX)
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; mov Rq(7), Rq(MEM)
        ; mov Rd(6), Rd(EFF)
        ; mov Rd(2), Rd(value)
        ; mov Rq(RAX), QWORD (func as i64)
        ; call Rq(RAX)
    );

    // Check for a host-reported access fault as soon as the callback returns: one `cmp` and one `jne`,
    // present only on accesses that **go through a callback** (a windowed inline is mapped by contract and is not checked).
    dynasm!(ops
        ; cmp DWORD [Rq(STATE) + FAULT_OFF], 0
        ; jne => fault
    );
}

/// Returns the host offset when an absolute guest address falls inside a window (a compile-time constant).
///
/// Such addresses come from PC-relative addressing (`ldr r0, [pc, #x]`), folded into constants at translation time,
/// so "is it out of bounds" is decidable at compile time too: on a hit one addressed access is emitted with no run-time check.
fn absolute_offset(window: FastWindow, addr: u32, size: MemSize) -> Option<i32> {
    let bytes = match size {
        MemSize::U8 => 1,
        MemSize::U16 => 2,
        MemSize::U32 => 4,
    };
    let offset = addr.checked_sub(window.base)? as usize;
    if offset.checked_add(bytes)? <= window.len && offset <= i32::MAX as usize {
        Some(offset as i32)
    } else {
        None
    }
}

/// Returns `len - size` (the largest legal offset) when the window can be inlined safely, otherwise `None`.
fn window_limit(window: FastWindow, size: MemSize) -> Option<u32> {
    let bytes = match size {
        MemSize::U8 => 1,
        MemSize::U16 => 2,
        MemSize::U32 => 4,
    };
    let limit = window.len.checked_sub(bytes)?;
    u32::try_from(limit).ok()
}

/// Inlines one access; outside a window or out of bounds it falls back to the callback.
///
/// The shape emitted (the hit path falls straight through, no jump):
///
/// ```text
/// mov  t0, EFF
/// sub  t0, base        ; omitted when base == 0
/// cmp  t0, limit
/// ja   slow
/// mov  dst, [WINDOW + t0]
/// jmp  done
/// slow:
///   <callback>
/// done:
/// ```
///
/// The slow path is unchanged, so behaviour "outside a window" is **byte for byte** what it is with no window at all
/// (`SimpleMemory`'s out-of-range read-as-0 and dropped writes both come from the callback).
fn emit_read_value(ops: &mut Asm, funcs: &MemFuncs, fault: dynasmrt::DynamicLabel, size: MemSize) {
    // Compare segment by segment: a hit on any one of them is inlined.
    //
    // **Segment 0 uses `WINDOW` directly** (the prologue already loaded it, so a hit costs nothing); other segments load their base into
    // `T1`. `WINDOW` must never be repointed at the segment that hit: [`emit_load`]'s "constant address" arm
    // accesses through `[WINDOW + a compile-time offset]`, and that offset was computed against **segment 0**;
    // once `WINDOW` points at another segment, later constant-address accesses in the same block read a different stretch of memory
    // (`tests/multiwindow.rs` has exactly this shape: hit segment 1 first, then take a constant address).
    let mut hits: Vec<(dynasmrt::DynamicLabel, u8, u8, u64)> = Vec::new();
    for (index, window) in funcs.windows.iter().enumerate() {
        let Some(window) = *window else { continue };
        let Some(limit) = window_limit(window, size) else {
            continue;
        };
        let fast = ops.new_dynamic_label();
        // With a zero window base the address is the offset already, so no move is needed (`SimpleMemory` is this kind)
        let offset = if window.base == 0 {
            EFF
        } else {
            dynasm!(ops ; mov Rd(T0), Rd(EFF) ; sub Rd(T0), window.base as i32);
            T0
        };
        dynasm!(ops
            ; cmp Rd(offset), limit as i32
            ; jbe => fast
        );
        hits.push((fast, offset, window_reg(index), window.ptr as u64));
    }
    if hits.is_empty() {
        return emit_call_read(ops, funcs.read[size_index(size)], fault);
    }
    let done = ops.new_dynamic_label();
    // The slow path sits to the side: a hit never reaches it, and it costs no unconditional jump
    emit_call_read(ops, funcs.read[size_index(size)], fault);
    dynasm!(ops ; jmp => done);
    for (fast, offset, window, ptr) in hits {
        dynasm!(ops ; => fast);
        if window != WINDOW {
            dynasm!(ops ; mov Rq(window), QWORD ptr as i64);
        }
        match size {
            MemSize::U8 => dynasm!(ops ; movzx Rd(RESULT), BYTE [Rq(window) + Rq(offset)]),
            MemSize::U16 => dynasm!(ops ; movzx Rd(RESULT), WORD [Rq(window) + Rq(offset)]),
            MemSize::U32 => dynasm!(ops ; mov Rd(RESULT), [Rq(window) + Rq(offset)]),
        }
        dynasm!(ops ; jmp => done);
    }
    dynasm!(ops ; => done);
}

/// The host register that serves as the base when inlining segment `index`.
///
/// The prologue puts segment 0's base into `WINDOW` and it **never changes**, so a hit needs no reload;
/// other segments use `T1` ([`EFF`] holds the offset, and `T1`'s old value is needed neither before nor after the access).
#[inline]
fn window_reg(index: usize) -> u8 {
    if index == 0 { WINDOW } else { T1 }
}

/// Writes `value` to `[EFF]`, inlined on a window hit and through the callback otherwise. See [`emit_read_value`] for the shape.
fn emit_write_value(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    size: MemSize,
    value: u8,
) {
    let mut hits: Vec<(dynasmrt::DynamicLabel, u8, u8, u64)> = Vec::new();
    for (index, window) in funcs.windows.iter().enumerate() {
        let Some(window) = *window else { continue };
        let Some(limit) = window_limit(window, size) else {
            continue;
        };
        let fast = ops.new_dynamic_label();
        let offset = if window.base == 0 {
            EFF
        } else {
            dynasm!(ops ; mov Rd(T0), Rd(EFF) ; sub Rd(T0), window.base as i32);
            T0
        };
        dynasm!(ops
            ; cmp Rd(offset), limit as i32
            ; jbe => fast
        );
        hits.push((fast, offset, window_reg(index), window.ptr as u64));
    }
    if hits.is_empty() {
        return emit_call_write(ops, funcs.write[size_index(size)], value, fault);
    }
    let done = ops.new_dynamic_label();
    emit_call_write(ops, funcs.write[size_index(size)], value, fault);
    dynasm!(ops ; jmp => done);
    for (fast, offset, window, ptr) in hits {
        dynasm!(ops ; => fast);
        if window != WINDOW {
            dynasm!(ops ; mov Rq(window), QWORD ptr as i64);
        }
        match size {
            MemSize::U8 => dynasm!(ops ; mov BYTE [Rq(window) + Rq(offset)], Rb(value)),
            MemSize::U16 => dynasm!(ops ; mov WORD [Rq(window) + Rq(offset)], Rw(value)),
            MemSize::U32 => dynasm!(ops ; mov [Rq(window) + Rq(offset)], Rd(value)),
        }
        dynasm!(ops ; jmp => done);
    }
    dynasm!(ops ; => done);
}

#[allow(clippy::too_many_arguments)]
fn emit_load(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    dst: u8,
    address: Address,
    size: MemSize,
    signed: bool,
    pre_index: bool,
    writeback: bool,
) {
    // An absolute address (a constant folded out of PC-relative addressing): whether a window covers it is known at compile time,
    // so on a hit not even a bounds check is emitted.
    if let (Address::Absolute(addr), Some(window)) = (address, funcs.window())
        && let Some(offset) = absolute_offset(window, addr, size)
    {
        match size {
            MemSize::U8 if signed => dynasm!(ops ; movsx Rd(RESULT), BYTE [Rq(WINDOW) + offset]),
            MemSize::U16 if signed => {
                dynasm!(ops ; movsx Rd(RESULT), WORD [Rq(WINDOW) + offset])
            }
            MemSize::U8 => dynasm!(ops ; movzx Rd(RESULT), BYTE [Rq(WINDOW) + offset]),
            MemSize::U16 => dynasm!(ops ; movzx Rd(RESULT), WORD [Rq(WINDOW) + offset]),
            MemSize::U32 => dynasm!(ops ; mov Rd(RESULT), [Rq(WINDOW) + offset]),
        }
        store_guest(ops, dst, RESULT);
        return;
    }

    let writeback_info = emit_address(ops, address, pre_index);
    if writeback && let Some((base, offset_addr)) = writeback_info {
        store_guest(ops, base, offset_addr);
    }

    emit_read_value(ops, funcs, fault, size);

    if signed {
        match size {
            MemSize::U8 => dynasm!(ops ; movsx Rd(RESULT), Rb(RESULT)),
            MemSize::U16 => dynasm!(ops ; movsx Rd(RESULT), Rw(RESULT)),
            MemSize::U32 => {}
        }
    }
    store_guest(ops, dst, RESULT);
}

#[allow(clippy::too_many_arguments)]
fn emit_store(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    src: u8,
    address: Address,
    size: MemSize,
    pre_index: bool,
    writeback: bool,
) {
    // Take the value first: if src happens to be the base register, the writeback would overwrite it
    load_guest(ops, T3, src);

    if let (Address::Absolute(addr), Some(window)) = (address, funcs.window())
        && let Some(offset) = absolute_offset(window, addr, size)
    {
        match size {
            MemSize::U8 => dynasm!(ops ; mov BYTE [Rq(WINDOW) + offset], Rb(T3)),
            MemSize::U16 => dynasm!(ops ; mov WORD [Rq(WINDOW) + offset], Rw(T3)),
            MemSize::U32 => dynasm!(ops ; mov [Rq(WINDOW) + offset], Rd(T3)),
        }
        return;
    }

    let writeback_info = emit_address(ops, address, pre_index);
    if writeback && let Some((base, offset_addr)) = writeback_info {
        store_guest(ops, base, offset_addr);
    }

    emit_write_value(ops, funcs, fault, size, T3);
}

/// Computes the block's starting address into the spill area. An `LDM`/`STM` transfer address advances per iteration,
/// and the address register is volatile (destroyed by every callback), so the cursor lives on the stack.
fn emit_block_address(ops: &mut Asm, base: u8, regs: u16, add: bool, pre_index: bool) {
    load_guest(ops, ADDR_TMP, base);
    let total = regs.count_ones() as i32 * 4;
    let bias = match (pre_index, add) {
        (false, true) => 0,
        (true, true) => 4,
        (false, false) => -(total - 4),
        (true, false) => -total,
    };
    if bias != 0 {
        dynasm!(ops ; add Rd(ADDR_TMP), bias);
    }
    dynasm!(ops ; mov [Rq(4) + SPILL_OFF], Rd(ADDR_TMP));
}

fn emit_block_writeback(ops: &mut Asm, base: u8, regs: u16, add: bool) {
    load_guest(ops, ADDR_TMP, base);
    let total = regs.count_ones() as i32 * 4;
    if add {
        dynasm!(ops ; add Rd(ADDR_TMP), total);
    } else {
        dynasm!(ops ; sub Rd(ADDR_TMP), total);
    }
    store_guest(ops, base, ADDR_TMP);
}

#[allow(clippy::too_many_arguments)]
/// **Whole-range window inlining** for register-list transfers (LDM/STM).
///
/// The per-register path costs three things per register: cursor traffic through the spill slot (3 instructions),
/// a per-segment window check (~4 instructions per window), and the access itself. But a transfer's whole address range
/// is known at compile time (`first + 4*i`), so as long as the **whole range** falls inside one segment the check
/// can be hoisted before the transfer, after which each register needs only one displaced addressed `mov`
/// (`[WINDOW + T0 + 4*i]`, x86's SIB addressing taking base+index+disp8 directly):
///
/// ```text
/// mov  EFF, [rsp + SPILL_OFF]      ; the first transfer address (computed by emit_block_address)
/// sub  EFF, w.base                 ; relative to the window (unsigned underflow is caught by the 64-bit check)
/// mov  T0, EFF                     ; zero-extended
/// add  T0, total - 4               ; no 64-bit wrap: the offset of the last access
/// cmp  T0, w.limit                 ; is the whole range inside the window (the largest legal U32 offset)
/// ja   next_window
/// mov  BASE, w.ptr                 ; needed only for a segment other than 0 (the prologue loaded that one)
/// ;   per register: mov RESULT, [BASE + T0 + 4*i] + store_guest (LDM)
/// ;              load_guest + mov [BASE + T0 + 4*i]          (STM)
/// jmp  merge
/// next_window: … (the next window; falling through into the per-register slow path)
/// ```
///
/// The fast path makes no callback (the window contract guarantees no missing page and no reported fault), so keeping T0 resident
/// in a register is safe; a miss (the range spans windows, or there is none) takes the original per-register path with byte-for-byte
/// identical semantics. A base inside the transfer list (`LDM r0, {r0, …}`) does not matter either, since the address sequence uses the snapshot
/// of the first address in the spill slot rather than following the loaded values. The 64-bit addition is what turns "the start lies below the window"
/// underflow into a huge unsigned number that one `ja` rejects.
///
/// The base register has to be chosen per segment ([`window_reg`]): what `WINDOW` holds from segment 0 onwards is
/// **segment 0's** host pointer, while the offset is computed against the segment that actually hit. Hitting a later segment but reusing
/// `WINDOW` would read and write the whole transfer at that offset in segment 0 — a single-register LDM reading 0
/// would send the `CMP/BEQ` that follows down the wrong path. The same convention is used here as in `emit_read_value`:
/// segment 0 uses `WINDOW` directly, and other segments load `w.ptr` into `T1`.
///
/// Returns whether the fast path was emitted (the caller uses that to skip the per-register path).
fn emit_run_fast(
    ops: &mut Asm,
    funcs: &MemFuncs,
    regs: &[u8],
    total: i32,
    load: bool,
    merge: dynasmrt::DynamicLabel,
) -> bool {
    let mut any_window = false;
    for (index, window) in funcs.windows.iter().enumerate() {
        let Some(window) = *window else { continue };
        let Some(w_limit) = window_limit(window, MemSize::U32) else {
            continue;
        };
        any_window = true;
        let base_reg = window_reg(index);
        let next = ops.new_dynamic_label();
        // A 32-bit mov zero-extends: T0's 64-bit value is the window-relative start. If the start lies below the window,
        // the subtraction wraps into a huge unsigned number, the 64-bit addition does not wrap, and one `ja` rejects it.
        dynasm!(ops
            ; mov Rd(EFF), [Rq(4) + SPILL_OFF]
            ; sub Rd(EFF), window.base as i32
            ; mov Rd(T0), Rd(EFF)
            ; add Rq(T0), (total - 4)
            ; cmp Rq(T0), w_limit as i32
            ; ja =>next
            ; mov Rd(T0), Rd(EFF)
        );
        // Only the hit path loads the base, and `WINDOW` must not be changed (constant-address accesses in the same block are based on it).
        if base_reg != WINDOW {
            dynasm!(ops ; mov Rq(base_reg), QWORD window.ptr as i64);
        }
        for (i, reg) in regs.iter().enumerate() {
            let disp = (i * 4) as i32;
            if load {
                dynasm!(ops ; mov Rd(RESULT), [Rq(base_reg) + Rq(T0) + disp]);
                store_guest(ops, *reg, RESULT);
            } else {
                load_guest(ops, RESULT, *reg);
                dynasm!(ops ; mov [Rq(base_reg) + Rq(T0) + disp], Rd(RESULT));
            }
        }
        dynasm!(ops
            ; jmp => merge
            ; =>next
        );
    }
    any_window
}

#[allow(clippy::too_many_arguments)]
fn emit_load_multiple(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    base: u8,
    regs: u16,
    add: bool,
    pre_index: bool,
    writeback: bool,
) {
    emit_block_address(ops, base, regs, add, pre_index);
    let ordered: Vec<u8> = (0..16u8).filter(|i| regs & (1 << i) != 0).collect();
    let total = ordered.len() as i32 * 4;
    let merge = ops.new_dynamic_label();
    // The hit path jumps straight to merge inside emit_run_fast; if the range does not fit it falls into the slow path below.
    emit_run_fast(ops, funcs, &ordered, total, true, merge);
    for index in 0..16u8 {
        if regs & (1 << index) == 0 {
            continue;
        }
        dynasm!(ops ; mov Rd(EFF), [Rq(4) + SPILL_OFF]);
        emit_read_value(ops, funcs, fault, MemSize::U32);
        store_guest(ops, index, RESULT);
        dynasm!(ops
            ; mov Rd(T0), [Rq(4) + SPILL_OFF]
            ; add Rd(T0), 4
            ; mov [Rq(4) + SPILL_OFF], Rd(T0)
        );
    }
    dynasm!(ops ; => merge);
    if writeback {
        emit_block_writeback(ops, base, regs, add);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_store_multiple(
    ops: &mut Asm,
    funcs: &MemFuncs,
    fault: dynasmrt::DynamicLabel,
    base: u8,
    regs: u16,
    add: bool,
    pre_index: bool,
    writeback: bool,
) {
    emit_block_address(ops, base, regs, add, pre_index);
    let ordered: Vec<u8> = (0..16u8).filter(|i| regs & (1 << i) != 0).collect();
    let total = ordered.len() as i32 * 4;
    let merge = ops.new_dynamic_label();
    emit_run_fast(ops, funcs, &ordered, total, false, merge);
    for index in 0..16u8 {
        if regs & (1 << index) == 0 {
            continue;
        }
        load_guest(ops, T3, index);
        dynasm!(ops ; mov Rd(EFF), [Rq(4) + SPILL_OFF]);
        emit_write_value(ops, funcs, fault, MemSize::U32, T3);
        dynasm!(ops
            ; mov Rd(T0), [Rq(4) + SPILL_OFF]
            ; add Rd(T0), 4
            ; mov [Rq(4) + SPILL_OFF], Rd(T0)
        );
    }
    dynasm!(ops ; => merge);
    if writeback {
        emit_block_writeback(ops, base, regs, add);
    }
}
