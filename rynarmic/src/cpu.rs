//! The CPU shell: fetch, decode, execute, advance PC.
//!
//! Two paths are offered: `step` decodes and executes one instruction at a time, while `run_cached`
//! goes through the basic-block cache and prefers JIT code. A cache miss translates a block; a block
//! that cannot be compiled falls back to the IR interpreter.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::{self, CodeArena, ExchangeSlot, JitBlock, JitFn};
use crate::block::{self, BasicBlock, Translation};
use crate::common::bits::cond_holds;
use crate::decoder::{self, DecodeError};
use crate::executor::{Executor, Flow};
use crate::ir::IrOp;
use crate::memory::{AccessFault, AccessKind, Memory};
use crate::state::ArmState;

/// Something that happened during execution and needs the host's attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionKind {
    UndefinedInstruction(u32),
    UnsupportedInstruction(DecodeError),
    SoftwareInterrupt(u32),
    Breakpoint(u32),
    /// An access or a fetch reached an address the host cannot serve (reported through
    /// [`Memory::set_fault_flag`]).
    ///
    /// `addr` and `kind` come from [`Memory::take_fault`]; the caller reads the PC at the point of
    /// failure from `state.pc()`. The interpreter is exact to the instruction; the JIT path is exact
    /// to the **block start**, since a block keeps no per-instruction PC — the same convention
    /// dynarmic uses.
    AccessFault {
        addr: u32,
        kind: AccessKind,
    },
}

impl core::fmt::Display for ExceptionKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExceptionKind::UndefinedInstruction(enc) => {
                write!(f, "undefined instruction at {enc:#010x}")
            }
            ExceptionKind::UnsupportedInstruction(err) => write!(f, "{err}"),
            ExceptionKind::SoftwareInterrupt(imm) => write!(f, "swi #{imm:#x}"),
            ExceptionKind::Breakpoint(imm) => write!(f, "bkpt #{imm:#x}"),
            ExceptionKind::AccessFault { addr, kind } => {
                write!(
                    f,
                    "{} fault at {addr:#010x}",
                    match kind {
                        AccessKind::Fetch => "prefetch",
                        AccessKind::Read => "read",
                        AccessKind::Write => "write",
                    }
                )
            }
        }
    }
}

impl std::error::Error for ExceptionKind {}

/// The result of a single step, for the caller to observe (statistics, breakpoints, and so on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    /// An instruction executed normally.
    Executed,
    /// The instruction was skipped by its condition code, or was only the first half of a 32-bit
    /// Thumb instruction.
    Skipped,
}

/// How many slots the single-step path's decode cache holds.
///
/// Direct mapping is enough: the PCs the single-step path visits cluster in a small range (a
/// function body plus its loop), so 512 slots hold a whole stretch of hot code, and a collision only
/// degrades to decoding again.
const STEP_CACHE_SLOTS: usize = 512;

/// The decode result for one instruction, reused by the single-step path.
///
/// **The key includes the instruction word itself**, not just the address, so a guest that rewrites
/// its code invalidates the entry automatically, with no explicit invalidation call. `pc` is part of
/// the key too, because PC-relative addressing is folded into an absolute address at decode time.
struct StepEntry {
    valid: bool,
    pc: u32,
    inst: u32,
    /// bit0 = Thumb, bit1 = the condition held (both outcomes of a conditional execution are
    /// cached separately)
    flags: u8,
    /// What r15 reads during execution (ARM = pc+8, Thumb = (pc+4) & !3)
    pc_during: u32,
    next_pc: u32,
    ops: Vec<IrOp>,
}

impl StepEntry {
    fn new() -> Self {
        Self {
            valid: false,
            pc: 0,
            inst: 0,
            flags: 0,
            pc_during: 0,
            next_pc: 0,
            ops: Vec::new(),
        }
    }
}

/// The direct-mapped index into the decode cache.
///
/// Both the instruction word and the pc take part: the former makes an entry invalidate itself when
/// the code is rewritten, the latter keeps PC-relative addressing (folded into absolute addresses at
/// decode time) from landing on the wrong address.
fn step_slot(pc: u32, inst: u32, code: u8) -> usize {
    let mut h = pc.wrapping_mul(0x9E37_79B1) ^ inst ^ ((code as u32) << 24);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    ((h ^ (h >> 13)) as usize) & (STEP_CACHE_SLOTS - 1)
}

pub struct Cpu<M: Memory> {
    pub state: ArmState,
    pub memory: M,
    /// The first halfword of a Thumb `BL`/`BLX` (address, encoding), waiting for the second half to
    /// pair it with.
    pending_thumb_bl: Option<(u32, u16)>,
    /// An IR buffer reused per instruction, so nothing is allocated per step.
    scratch: Vec<IrOp>,
    /// Whether to attempt compiling basic blocks to machine code.
    jit_enabled: bool,
    /// How many more new blocks may be compiled this turn; `None` means no limit.
    ///
    /// Compilation happens on the executing thread, so compiling a few thousand blocks in one turn
    /// keeps the host busy for hundreds of milliseconds **inside a single call**, with the picture
    /// and the state both frozen. When the quota runs out, the remaining blocks are still decoded but
    /// produce no machine code (`entry` stays null, so they run on the interpreter) and are filled in
    /// next turn when the budget refreshes — the cost of entering a new scene is spread over several
    /// frames that way. Unlike `jit_enabled`, changing this does **not** clear the cache: it affects
    /// only future translations, and already-compiled blocks keep running.
    compile_budget: Option<u32>,
    /// Whether to inline accesses inside a window as host `mov`s (P1 in `docs/EFFICIENCY.md`).
    ///
    /// The window base is an immediate baked into the machine code, so toggling this has to clear
    /// the cache.
    fastmem_enabled: bool,
    /// The previous block's entry, cached.
    ///
    /// In a tight loop (`bne` back to the block start) the PC is identical every round, so comparing
    /// it hits directly and saves the chain of dependent loads that computing the index, reading the
    /// slot and comparing the tag would cost.
    last_block: Option<LastBlock>,
    /// Owner of the compiled code.
    ///
    /// An entry address may not become invalid when its cache slot is overwritten: `last_block` can
    /// still point at this code after the slot was taken by another address, so ownership of the
    /// code lives here and the cache slot records only the entry address. The machine code itself
    /// lives in [`Core::code_arena`]'s mapping, so `Vec` reallocation moving the `JitBlock`
    /// structures does not invalidate it.
    jit_arena: Vec<JitBlock>,
    /// The machine code itself: one block of memory appended to block by block.
    ///
    /// Once every block carried its own `mmap` plus `mprotect`, which was 62% of the per-block
    /// compilation cost (52 µs). With "a few mappings per process, everything else a byte copy" the
    /// cost of producing the same code fell from 35.2 µs to 0.82 µs.
    ///
    /// It lives as long as `jit_arena`: `invalidate_code_cache` clears the blocks first and only then
    /// rewinds the cursor.
    code_arena: CodeArena,
    /// The compiled-code ceiling; tests lower it to exercise the invalidation path.
    jit_arena_limit: usize,
    /// Every backpatchable link slot, registered across blocks.
    ///
    /// The slots live in their own blocks' heap allocations, so this table has to be cleared
    /// **before** `jit_arena`: on invalidation the slots are written back to "unlinked" while the
    /// code pages are still there to write to.
    link_sites: Vec<LinkSite>,
    /// Indices into `link_sites`, bucketed by "target (pc, thumb, source frame layout)".
    ///
    /// Without it `patch_links` would scan **every** link site for each block compiled, making
    /// compiling N blocks O(N²) — around 400 million comparisons at 20,000 blocks, a measurable
    /// share of cold compilation. The index is used only while compiling and never enters the hot
    /// path. The `frame` in the key is the **source** block's frame layout, because a backpatch
    /// happens only on edges where both sides lay out the frame the same way (a link jump skips the
    /// prologue).
    link_index: HashMap<(u32, bool, bool), Vec<usize>>,
    /// The single-step path's decode cache. Allocated lazily: a host that never calls `step()` does
    /// not pay for the memory.
    step_cache: Vec<StepEntry>,
    /// The direct-mapped basic-block cache: locate with `(pc >> 1) & MASK`, then compare the tag
    /// inside the slot. Direct mapping rather than a hash table, because a typical block is only 2 to
    /// 10 instructions and hashing would eat the gain.
    blocks: Vec<Option<BlockSlot>>,
}

/// One link site waiting to be backpatched.
#[derive(Clone, Copy)]
struct LinkSite {
    /// The slot's heap address, referenced as an immediate from the machine code.
    slot: *const AtomicU64,
    /// The target while unlinked (the source block's epilogue).
    stub: u64,
    /// The successor block's guest address and instruction set.
    target_pc: u32,
    thumb: bool,
    /// The source block's frame layout class; only matching frames may link (see
    /// `emit_link_site`).
    frame: bool,
}

/// What the driver has to remember after a successful compilation.
#[derive(Clone, Copy)]
struct CompiledEntry {
    /// The function entry (prologue).
    entry: *const u8,
    /// The block body's entry (past the prologue), where a link jump lands.
    body: *const u8,
    /// The frame layout class.
    frame: bool,
    /// The inline-cache slot of an exchange terminator.
    exchange: *const ExchangeSlot,
}

impl CompiledEntry {
    const NONE: CompiledEntry = CompiledEntry {
        entry: std::ptr::null(),
        body: std::ptr::null(),
        frame: false,
        exchange: std::ptr::null(),
    };
}

/// The result of one block lookup: everything needed to run a block.
#[derive(Clone, Copy)]
struct Loaded {
    /// The machine-code entry (prologue); null means there is no JIT version and the interpreter
    /// runs it.
    entry: *const u8,
    /// The block body's entry (past the prologue), where static links and the inline cache land.
    body: *const u8,
    /// The frame layout class: a jump into a body is only allowed between matching frames.
    frame: bool,
    /// The inline-cache slot of an exchange terminator; null means this block has no exchange
    /// branch.
    exchange: *const ExchangeSlot,
}

impl Loaded {
    /// Interpreted only: there is no JIT version.
    const INTERPRETED: Loaded = Loaded {
        entry: std::ptr::null(),
        body: std::ptr::null(),
        frame: false,
        exchange: std::ptr::null(),
    };
}

/// The fast-hit information for the previous block.
#[derive(Clone, Copy)]
struct LastBlock {
    pc: u32,
    thumb: bool,
    loaded: Loaded,
}

/// The ceiling on compiled code: on reaching it everything is invalidated, so the code does not
/// pile up indefinitely under heavy churn.
const JIT_ARENA_LIMIT: usize = 16384;

struct BlockSlot {
    tag: u64,
    /// The compiled entry; null means this block has no JIT version and runs on the interpreter.
    ///
    /// Sitting first in the structure is deliberate: the hot path reads only `tag`/`entry`/`count`,
    /// and those three fields share one cache line, so running a block does not have to chase
    /// `BasicBlock`'s heap pointer.
    entry: *const u8,
    /// How many guest instructions the block holds.
    count: u32,
    /// The sequential exit. Needed only by the interpreter path and for locating an exception.
    next_pc: u32,
    /// The compiled body's entry (past the prologue).
    ///
    /// A link site may be registered **after** its target block was compiled (which is what happens
    /// when a `b` jumps backwards), and by then there is no "a new block was compiled" event to
    /// trigger the backpatch, so registration also looks the cache slot up.
    body: *const u8,
    /// The body's frame layout class, compared against a link site when backpatching.
    frame: bool,
    /// The inline-cache slot of an exchange terminator (null means this block has no exchange
    /// branch).
    exchange: *const ExchangeSlot,
    /// `None` means this address cannot form a block and has to be single-stepped.
    ///
    /// Caching a failed translation matters: a translation allocates a `Vec`, and such an address is
    /// retried every time it is passed. Uncached, code dense with conditional branches measured
    /// 3.8 MIPS — seven times slower than plain single-stepping.
    block: Option<Box<BasicBlock>>,
    /// This block could have been compiled, but the turn's budget was exhausted (see
    /// [`Cpu::set_compile_budget`]).
    ///
    /// It has to be recorded: once the budget refreshes, a retranslation has to be allowed, or the
    /// block gets cached as "this address cannot be compiled" forever and that block runs on the
    /// interpreter from then on.
    denied: bool,
}

impl BlockSlot {
    /// The cache entry for "this address can only be single-stepped".
    fn single_step(tag: u64) -> BlockSlot {
        BlockSlot {
            tag,
            entry: std::ptr::null(),
            count: 0,
            next_pc: 0,
            body: std::ptr::null(),
            frame: false,
            exchange: std::ptr::null(),
            block: None,
            denied: false,
        }
    }
}

/// How many cache slots there are (a power of two).
///
/// Indexing on the low bits alone would cover only `BLOCK_CACHE_SIZE * 2` bytes of address space:
/// 8 KiB at 4096 slots, while a real application image is hundreds of KiB, so two hot stretches
/// `0x14000` or `0x42000` apart would land in exactly the same slot. Under direct mapping they evict
/// each other every round, and the same blocks are translated and compiled over and over —
/// measured, 300 ms of virtual time produced 7911 compilations for only 2155 distinct blocks, the
/// hottest pair (0x10039388 / 0x10025388) recompiling 68 times each, leaving the JIT 2.4 times
/// slower than the interpreter. 65536 slots cover 128 KiB and, with the high-bit folding below, hot
/// block collisions disappear.
const BLOCK_CACHE_SIZE: usize = 65536;
/// `log2(BLOCK_CACHE_SIZE)`, used for the folding; change it along with the slot count.
const CACHE_INDEX_BITS: u32 = 16;

#[inline]
fn cache_index(pc: u32, thumb: bool) -> usize {
    // Thumb instructions are halfword-aligned and ARM word-aligned, so shift both right by one.
    // Taking only the low bits is the same as "address modulo 128 KiB" — precisely the stride
    // compiler layout produces most often — so the high bits are xor-ed in and the whole address
    // takes part in the decision. (A multiplicative hash will not do: an odd multiplier is a
    // bijection on the low sixteen bits, so no high-bit information gets in at all.)
    let h = pc >> 1;
    let mixed = h ^ (h >> CACHE_INDEX_BITS) ^ (h >> 26);
    ((mixed as usize) ^ ((thumb as usize) << (CACHE_INDEX_BITS - 1))) & (BLOCK_CACHE_SIZE - 1)
}

#[inline]
fn cache_tag(pc: u32, thumb: bool) -> u64 {
    (pc as u64) | ((thumb as u64) << 32)
}

impl<M: Memory> Cpu<M> {
    pub fn new(memory: M) -> Self {
        Self {
            state: ArmState::default(),
            memory,
            pending_thumb_bl: None,
            scratch: Vec::with_capacity(4),
            last_block: None,
            jit_arena: Vec::new(),
            code_arena: backend::CodeArena::new(backend::MAX_ARENA_BYTES),
            jit_arena_limit: JIT_ARENA_LIMIT,
            link_sites: Vec::new(),
            link_index: HashMap::new(),
            jit_enabled: true,
            compile_budget: None,
            fastmem_enabled: true,
            step_cache: Vec::new(),
            // The cache slots are allocated only once `run_cached` is used, so a purely
            // single-stepping host does not waste the memory
            blocks: Vec::new(),
        }
    }

    /// Executes straight from the decode cache where possible; `None` means a miss.
    ///
    /// The single-step path re-decodes one instruction on every `step()` (the A32 decoder is a large
    /// match, measured at about 21 ns per instruction, two fifths of the total cost). Decoding the
    /// same encoding at the same address over and over is pure waste: hot loops, function bodies and
    /// frequently used system-call stubs all come back to the same handful of PCs.
    fn try_step_cache(
        &mut self,
        pc: u32,
        inst: u32,
        thumb: bool,
        passed: bool,
    ) -> Option<Result<StepOutcome, ExceptionKind>> {
        let code = (thumb as u8) | ((passed as u8) << 1);
        let slot = step_slot(pc, inst, code);
        let entry = self.step_cache.get(slot)?;
        if !entry.valid || entry.pc != pc || entry.inst != inst || entry.flags != code {
            return None;
        }
        let during = entry.pc_during;
        let next = entry.next_pc;
        let empty = entry.ops.is_empty();
        // Matching the decode path: r15 during execution is the pipelined value
        self.state.set_pc(during);
        let flow = {
            let ops = &self.step_cache[slot].ops;
            Executor::new(&mut self.state, &mut self.memory).execute(ops)
        };
        Some(self.finish(flow, empty, next, pc))
    }

    /// Stores the operations just decoded in the cache (lazily allocated, overwritten when full).
    fn fill_step_cache(
        &mut self,
        pc: u32,
        inst: u32,
        thumb: bool,
        passed: bool,
        pc_during: u32,
        next_pc: u32,
    ) {
        if self.step_cache.is_empty() {
            self.step_cache = (0..STEP_CACHE_SLOTS).map(|_| StepEntry::new()).collect();
        }
        let code = (thumb as u8) | ((passed as u8) << 1);
        let slot = step_slot(pc, inst, code);
        let (cache, scratch) = (&mut self.step_cache, &self.scratch);
        let entry = &mut cache[slot];
        entry.valid = true;
        entry.pc = pc;
        entry.inst = inst;
        entry.flags = code;
        entry.pc_during = pc_during;
        entry.next_pc = next_pc;
        entry.ops.clear();
        entry.ops.extend_from_slice(scratch);
    }

    /// Hands the access-fault flag slot to the memory implementation and clears it.
    ///
    /// Rebound before every run: the host may move the `Cpu`, so the slot address must not be cached
    /// across calls.
    fn bind_fault_flag(&mut self) {
        self.state.fault = 0;
        let slot = &mut self.state.fault as *mut u32;
        self.memory.set_fault_flag(slot);
    }

    /// Turns a host-reported access fault into an exception, clearing the flag on the way.
    ///
    /// With `as_fetch` the kind is rewritten to [`AccessKind::Fetch`]: a fetch goes through the
    /// ordinary `read_u32` on the host side, so the memory implementation cannot tell "fetch" from
    /// "data read" — only the call site knows.
    fn take_memory_fault(&mut self, as_fetch: bool) -> ExceptionKind {
        let fault = self.memory.take_fault().unwrap_or(AccessFault {
            addr: self.state.pc(),
            kind: AccessKind::Read,
        });
        self.state.fault = 0;
        ExceptionKind::AccessFault {
            addr: fault.addr,
            kind: if as_fetch {
                AccessKind::Fetch
            } else {
                fault.kind
            },
        }
    }

    /// Executes one instruction (A32 or Thumb, as CPSR.T decides).
    pub fn step(&mut self) -> Result<StepOutcome, ExceptionKind> {
        self.bind_fault_flag();
        let pc = self.state.pc();

        if self.state.thumb() {
            self.step_thumb(pc)
        } else {
            self.step_arm(pc)
        }
    }

    fn step_thumb(&mut self, pc: u32) -> Result<StepOutcome, ExceptionKind> {
        let inst = self.memory.read_u16(pc);
        if self.state.fault != 0 {
            let kind = self.take_memory_fault(true);
            self.state.set_pc(pc);
            return Err(kind);
        }
        let flags = self.state.cpsr;

        // 32-bit Thumb: the prefix was stored, and this halfword completes the instruction
        if let Some((prefix_pc, prefix)) = self.pending_thumb_bl.take() {
            let ops = decoder::thumb::decode_bl_pair(prefix, inst, prefix_pc)
                .map_err(ExceptionKind::UnsupportedInstruction)?;
            self.state.set_pc((pc.wrapping_add(4)) & !3);
            return self.finish_with(&ops, pc.wrapping_add(2), pc);
        }
        if decoder::thumb::is_bl_prefix(inst) {
            // The first half is only a prefix: record it and advance to the next halfword, with no
            // execution effect
            self.pending_thumb_bl = Some((pc, inst));
            self.state.set_pc(pc.wrapping_add(2));
            return Ok(StepOutcome::Skipped);
        }

        let passed = cond_holds(
            ((inst >> 8) & 0xF) as u32,
            flags.n(),
            flags.z(),
            flags.c(),
            flags.v(),
        );
        if let Some(result) = self.try_step_cache(pc, inst as u32, true, passed) {
            return result;
        }
        self.scratch.clear();
        decoder::thumb::decode_into(inst, pc, passed, &mut self.scratch)
            .map_err(ExceptionKind::UnsupportedInstruction)?;
        // In Thumb state reading PC gives (instruction address + 4) with bit 1 cleared
        let during = (pc.wrapping_add(4)) & !3;
        let next_pc = pc.wrapping_add(2);
        self.state.set_pc(during);
        self.fill_step_cache(pc, inst as u32, true, passed, during, next_pc);
        self.finish_scratch(next_pc, pc)
    }

    fn step_arm(&mut self, pc: u32) -> Result<StepOutcome, ExceptionKind> {
        let inst = self.memory.read_u32(pc);
        if self.state.fault != 0 {
            let kind = self.take_memory_fault(true);
            self.state.set_pc(pc);
            return Err(kind);
        }
        let flags = self.state.cpsr;
        let cond = inst >> 28;
        let passed = cond == 0xF || cond_holds(cond, flags.n(), flags.z(), flags.c(), flags.v());
        if let Some(result) = self.try_step_cache(pc, inst, false, passed) {
            return result;
        }
        self.scratch.clear();
        decoder::a32::decode_into(inst, pc, passed, &mut self.scratch)
            .map_err(ExceptionKind::UnsupportedInstruction)?;
        // In A32 state reading PC gives (instruction address + 8)
        let during = pc.wrapping_add(8);
        let next_pc = pc.wrapping_add(4);
        self.state.set_pc(during);
        self.fill_step_cache(pc, inst, false, passed, during, next_pc);
        self.finish_scratch(next_pc, pc)
    }

    fn finish_scratch(
        &mut self,
        next_pc: u32,
        instruction_pc: u32,
    ) -> Result<StepOutcome, ExceptionKind> {
        let empty = self.scratch.is_empty();
        let flow = {
            let state = &mut self.state;
            let memory = &mut self.memory;
            Executor::new(state, memory).execute(&self.scratch)
        };
        self.finish(flow, empty, next_pc, instruction_pc)
    }

    fn finish_with(
        &mut self,
        ops: &[IrOp],
        next_pc: u32,
        instruction_pc: u32,
    ) -> Result<StepOutcome, ExceptionKind> {
        let empty = ops.is_empty();
        let flow = {
            let state = &mut self.state;
            let memory = &mut self.memory;
            Executor::new(state, memory).execute(ops)
        };
        self.finish(flow, empty, next_pc, instruction_pc)
    }

    fn finish(
        &mut self,
        flow: Flow,
        empty: bool,
        next_pc: u32,
        instruction_pc: u32,
    ) -> Result<StepOutcome, ExceptionKind> {
        match flow {
            Flow::Next => {
                self.state.set_pc(next_pc);
                if empty {
                    Ok(StepOutcome::Skipped)
                } else {
                    Ok(StepOutcome::Executed)
                }
            }
            Flow::Branch(target) => {
                self.state.set_pc(target);
                Ok(StepOutcome::Executed)
            }
            Flow::Exchange(target) => {
                self.state.cpsr.set_t(target & 1 != 0);
                self.state.set_pc(target & !1);
                Ok(StepOutcome::Executed)
            }
            Flow::Exception(kind) => {
                // On an exception PC goes back to the failing instruction and the host decides the
                // vector and the mode switch
                self.pending_thumb_bl = None;
                self.state.set_pc(instruction_pc);
                Err(kind)
            }
        }
    }

    /// Turns the JIT backend on or off. Toggling clears the block cache (the slots hold compiled
    /// machine code).
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        if self.jit_enabled != enabled {
            self.jit_enabled = enabled;
            self.invalidate_code_cache();
        }
    }

    /// Sets the ceiling on compiled code, in blocks.
    ///
    /// Lowering it exercises the "arena full, so invalidate before compiling" path (the randomized
    /// differential tests use it to force invalidation), and it also serves a host with a small code
    /// budget. Toggling clears the cache, because the old blocks' entry addresses are about to
    /// become invalid.
    pub fn set_jit_arena_limit(&mut self, limit: usize) {
        self.jit_arena_limit = limit.max(1);
        self.invalidate_code_cache();
    }

    /// Turns inlined accesses on or off. On by default; with it off, every access goes through a
    /// host callback.
    ///
    /// The window base is an immediate compiled into the machine code, so toggling clears the code
    /// cache, for the same reason as [`Cpu::set_jit_enabled`].
    pub fn set_fastmem_enabled(&mut self, enabled: bool) {
        if self.fastmem_enabled != enabled {
            self.fastmem_enabled = enabled;
            self.invalidate_code_cache();
        }
    }

    /// Whether accesses inside a window are currently inlined.
    pub fn fastmem_enabled(&self) -> bool {
        self.fastmem_enabled
    }

    /// Whether the JIT is on.
    pub fn jit_enabled(&self) -> bool {
        self.jit_enabled
    }

    /// Sets how many more new blocks may be compiled this turn; `None` means no limit.
    ///
    /// Unlike [`Cpu::set_jit_enabled`] this does **not** clear the cache: the quota only decides
    /// whether new blocks may still be compiled, and already-compiled blocks keep running, so
    /// refreshing it per turn does not disturb the hot path. When the quota is exhausted, new blocks
    /// are still decoded and simply run on the interpreter, with compilation catching up next turn
    /// when the budget returns.
    pub fn set_compile_budget(&mut self, budget: Option<u32>) {
        self.compile_budget = budget;
    }

    /// How much compilation quota is left this turn (`None` means no limit).
    pub fn compile_budget(&self) -> Option<u32> {
        self.compile_budget
    }

    /// Whether a new block may be compiled now: the JIT is on and this turn's quota is not used up.
    ///
    /// Read-only; it does not spend quota. A slot that was blocked by the quota uses it to decide
    /// whether to retry.
    fn compile_allowed(&self) -> bool {
        self.jit_enabled
            && match self.compile_budget {
                None => true,
                Some(left) => left > 0,
            }
    }

    /// Takes one unit of compilation quota. `false` means this block is not compiled this turn.
    fn take_compile_budget(&mut self) -> bool {
        match self.compile_budget {
            None => true,
            Some(left) if left > 0 => {
                self.compile_budget = Some(left - 1);
                true
            }
            Some(_) => false,
        }
    }

    /// Clears the basic-block cache.
    ///
    /// A guest that writes to its own code has to call this, or the old blocks keep running.
    pub fn invalidate_code_cache(&mut self) {
        self.state.materialize_lazy_flags();
        // Writing the link slots back to "unlinked" is **defensive**: invalidation discards
        // everything, and the slots live in blocks that are being released, so nobody touches them
        // after the table is cleared and today the step is unobservable.
        // It stays because single-block invalidation or eviction would make a dangling jump the
        // hardest kind of crash to find, and the order may not be reversed either: unlink first,
        // release the code pages after.
        for site in &self.link_sites {
            // SAFETY: the slot is owned by a block in `jit_arena`, which has not been released yet.
            unsafe { (*site.slot).store(site.stub, Ordering::Relaxed) };
        }
        self.link_sites.clear();
        // The index holds indices into `link_sites`, meaningless once the table is empty.
        self.link_index.clear();
        for slot in &mut self.blocks {
            *slot = None;
        }
        self.last_block = None;
        // The code pages go with the arena; any entry address left behind would dangle
        self.jit_arena.clear();
        // The code memory is no longer a per-block mmap but is carved out of `code_arena`; every
        // block is gone, so rewinding the cursor is safe here. **The order may not be reversed**:
        // clear the blocks first, reset after.
        self.code_arena.reset();
    }

    /// How many basic blocks are cached (for tests and observation).
    pub fn cached_block_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|slot| matches!(slot, Some(slot) if slot.block.is_some()))
            .count()
    }

    /// How many blocks compiled to machine code successfully (for tests and observation).
    pub fn compiled_block_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|slot| matches!(slot, Some(slot) if !slot.entry.is_null()))
            .count()
    }

    /// How many link sites are actually linked (their slot no longer points at their own epilogue),
    /// for tests and observation.
    pub fn linked_site_count(&self) -> usize {
        self.link_sites
            .iter()
            .filter(|site| {
                // SAFETY: the slot is owned by a block in `jit_arena` and is not released while it
                // is still in the table.
                (unsafe { (*site.slot).load(Ordering::Relaxed) }) != site.stub
            })
            .count()
    }

    /// Registers a newly compiled block: collects its link slots and backpatches the slots aiming at
    /// it.
    fn register_compiled(&mut self, jit: JitBlock) -> CompiledEntry {
        let entry = jit.entry_ptr();
        let body = jit.body_ptr();
        let exchange = jit.exchange_ptr();
        let start_pc = jit.start_pc();
        let thumb = jit.thumb();
        let needs_memory = jit.needs_memory();

        let first_site = self.link_sites.len();
        for slot in jit.slots() {
            let index = self.link_sites.len();
            self.link_sites.push(LinkSite {
                slot: slot.cell_ptr(),
                stub: slot.stub,
                target_pc: slot.target_pc,
                thumb: slot.thumb,
                frame: needs_memory,
            });
            // Bucket by target, so `patch_links` need not scan the whole table.
            self.link_index
                .entry((slot.target_pc, slot.thumb, needs_memory))
                .or_default()
                .push(index);
        }

        self.jit_arena.push(jit);
        // The block's own link sites are in the table too: a self-loop (`b .`) links through this
        // very step.
        self.patch_links(start_pc, thumb, needs_memory, body);

        // The other direction: this block's link sites may point at blocks compiled **long ago**
        // (a `b` jumping backwards), and that edge never sees a "target compiled" event, so the
        // cache slot has to be consulted on the spot.
        for index in first_site..self.link_sites.len() {
            self.patch_site_from_cache(self.link_sites[index]);
        }

        CompiledEntry {
            entry,
            body,
            frame: needs_memory,
            exchange,
        }
    }

    /// Backpatches one link site from the cache slot (its target may have been compiled before the
    /// site was registered).
    fn patch_site_from_cache(&self, site: LinkSite) {
        let index = cache_index(site.target_pc, site.thumb);
        let Some(slot) = &self.blocks[index] else {
            return;
        };
        if slot.tag == cache_tag(site.target_pc, site.thumb)
            && !slot.body.is_null()
            && slot.frame == site.frame
        {
            // SAFETY: the slot is owned by a block in `jit_arena`, and `slot.body` belongs to that
            // arena as well.
            unsafe { (*site.slot).store(slot.body as u64, Ordering::Relaxed) };
        }
    }

    /// Points every link slot aiming at `start_pc` at `body`.
    ///
    /// Only source blocks with **the same frame layout** are backpatched: a link jump skips the
    /// target's prologue, so both sides must lay out the stack frame identically — otherwise the
    /// body would use an unsaved `MEM` or spill area, or the epilogue would pop under the wrong
    /// layout.
    fn patch_links(&mut self, start_pc: u32, thumb: bool, frame: bool, body: *const u8) {
        // Take only the bucket for this target whose source frame matches: each pc is compiled once,
        // so the bucket need not be kept afterwards. Link sites registered later that also aim at
        // this pc are handled by the reverse backpatch in `register_compiled`.
        let Some(bucket) = self.link_index.remove(&(start_pc, thumb, frame)) else {
            return;
        };
        for index in bucket {
            let site = self.link_sites[index];
            // SAFETY: the slot is owned by a block in `jit_arena`, and `body` belongs to a block in
            // the same arena.
            unsafe { (*site.slot).store(body as u64, Ordering::Relaxed) };
        }
    }

    /// The execution loop built on the basic-block cache.
    ///
    /// Compared with [`Cpu::run`], nothing is decoded instruction by instruction inside a block and
    /// there is no PC dispatch per instruction.
    ///
    /// **The granularity is the block, not the instruction**: on return the number of instructions
    /// actually executed may slightly exceed `max_instructions` (by at most one block). That matches
    /// the interrupt check granularity of a real emulator, and a caller should service timers and
    /// interrupts at block boundaries. Use [`Cpu::step`] where exact single-stepping is required.
    pub fn run_cached(&mut self, max_instructions: u64) -> Result<u64, ExceptionKind> {
        let (executed, result) = self.run_cached_counted(max_instructions);
        result.map(|()| executed)
    }

    /// As [`Cpu::run_cached`], but **bringing back the executed count even when an exception
    /// interrupts it**.
    ///
    /// Benchmarks need that: letting the measured program end with a `SWI` (see
    /// `tests/support/prog/start.s`) gives both the "it finished" signal and an exact instruction
    /// count — no spinning in a dead loop, so idling never counts towards throughput.
    pub fn run_cached_counted(
        &mut self,
        max_instructions: u64,
    ) -> (u64, Result<(), ExceptionKind>) {
        let mut executed = 0u64;
        self.bind_fault_flag();

        if self.blocks.is_empty() {
            self.blocks = (0..BLOCK_CACHE_SIZE).map(|_| None).collect();
        }

        // PC is carried in a local around the loop and written back to the state only when needed: a
        // JIT block hands the new PC to the next round as its return value rather than through
        // memory, which removes one store-to-load dependency chain from the hot loop.
        self.state.materialize_lazy_flags();
        let mut pc = self.state.pc();
        // The inline-cache slot of the previous block (if it ended in an exchange branch) plus its
        // frame class: once the target block is known, `(pc, body)` goes in, so the next exchange at
        // the same place jumps straight there.
        let mut pending: Option<(*const ExchangeSlot, bool)> = None;

        while executed < max_instructions {
            let thumb = self.state.thumb();

            // The tight-loop fast path: neither the PC nor the instruction-set state changed, so the
            // previous block's entry is reused directly.
            //
            // No instruction count is needed here: the JIT reports what it executed through the
            // difference in `state.jit_budget`, and the interpreter path takes `count` from the cache
            // slot itself.
            let loaded = match self.last_block {
                Some(last) if last.pc == pc && last.thumb == thumb => last.loaded,
                _ => {
                    match self.lookup_or_translate(pc, thumb) {
                        Ok(result) => {
                            self.last_block = if result.entry.is_null() {
                                None
                            } else {
                                Some(LastBlock {
                                    pc,
                                    thumb,
                                    loaded: result,
                                })
                            };
                            result
                        }
                        Err(kind) => {
                            // On failure PC has to stop at **the failing instruction**, not at the
                            // previous block boundary
                            self.state.materialize_lazy_flags();
                            self.state.set_pc(pc);
                            return (executed, Err(kind));
                        }
                    }
                }
            };

            // Fill the inline cache: the source block just finished and the target is known. Blocks
            // with different frame layouts cannot jump into each other's bodies, so that case is left
            // empty (the stale value in the slot will not match, costing one extra compare).
            if let Some((slot, src_frame)) = pending.take()
                && !loaded.body.is_null()
                && loaded.frame == src_frame
            {
                // SAFETY: the slot is owned by a block in `jit_arena`; `loaded.body` belongs to the
                // same arena, and the arena is discarded whole (`invalidate_code_cache`), so no
                // target can dangle.
                unsafe { (*slot).store(pc, loaded.body) };
            }

            let entry = loaded.entry;
            if !entry.is_null() {
                // SAFETY: `entry` was produced by `compile::<M>` for the same M as this `Cpu<M>`;
                // the code is owned by `jit_arena` and stays valid until `invalidate_code_cache`;
                // the machine code touches only `state` and `memory`, and finishes before returning.
                let run: JitFn = unsafe { std::mem::transmute(entry) };
                // Hand over the budget left in this call; at the block tail the chain decides for
                // itself whether there is budget to keep chaining, which is how a chain cannot walk
                // around `max_instructions`.
                let remaining = (max_instructions - executed).min(u32::MAX as u64) as u32;
                self.state.jit_budget = remaining;
                pc = unsafe {
                    run(
                        &mut self.state as *mut _,
                        &mut self.memory as *mut _ as *mut u8,
                    )
                };
                // A chain may have run several blocks; it writes the remaining budget back to the
                // state, and the difference is what actually executed. The last block tests the
                // budget only after running whole, so the difference may exceed `remaining` by one
                // block.
                let consumed = i64::from(remaining) - i64::from(self.state.jit_budget as i32);
                executed += consumed.max(1) as u64;
                // For a block ending in an exchange branch, the returned `pc` is its branch target;
                // once the next round has looked up the block at that target, the slot is filled
                // (and a hit never comes back to the driver again).
                if !loaded.exchange.is_null() {
                    pending = Some((loaded.exchange, loaded.frame));
                }
                if self.state.fault != 0 {
                    // The block stopped at the fault (a block keeps no per-instruction PC, so PC
                    // reports the block start)
                    let kind = self.take_memory_fault(false);
                    self.state.materialize_lazy_flags();
                    self.state.set_pc(pc);
                    return (executed, Err(kind));
                }
                if self.state.pending_exception != 0 {
                    // SVC/BKPT: the marker a block raises. RESULT is already this instruction's
                    // address (pc is it), matching the interpreter's convention (PC stops at the SVC
                    // itself and the host advances it once it has been served).
                    let marker = std::mem::take(&mut self.state.pending_exception);
                    let kind = if marker & 0x4000_0000 != 0 {
                        ExceptionKind::Breakpoint(marker & 0x00FF_FFFF)
                    } else {
                        ExceptionKind::SoftwareInterrupt(marker & 0x00FF_FFFF)
                    };
                    self.state.materialize_lazy_flags();
                    self.state.set_pc(pc);
                    return (executed, Err(kind));
                }
                continue;
            }

            // The interpreter path reads PC and CPSR from the state, so sync them and materialize the
            // flags here
            self.state.materialize_lazy_flags();
            self.state.set_pc(pc);
            if let Err(kind) = self.run_interpreted(pc, thumb, &mut executed) {
                return (executed, Err(kind));
            }
            pc = self.state.pc();
        }

        self.state.materialize_lazy_flags();
        self.state.set_pc(pc);
        (executed, Ok(()))
    }

    /// Looks the block up and, on a miss, translates and (tries to) compile it. Returns the entry
    /// address and instruction count; an empty entry means it can only be single-stepped.
    fn lookup_or_translate(&mut self, pc: u32, thumb: bool) -> Result<Loaded, ExceptionKind> {
        let index = cache_index(pc, thumb);
        let tag = cache_tag(pc, thumb);

        {
            // A block blocked by the quota counts as a hit: it is already decoded and runs
            // interpreted, it just has no machine code yet. But once the quota is back it has to be
            // treated as a miss so it can catch up on compilation — otherwise "not compiled this
            // turn" would turn into "never compiled".
            let hit = matches!(
                &self.blocks[index],
                Some(slot) if slot.tag == tag && !(slot.denied && self.compile_allowed())
            );
            if !hit {
                let translated = block::translate(&mut self.memory, pc, thumb);
                if self.state.fault != 0 {
                    // A fetch hit an unmapped address: what the translation read is garbage and must
                    // not be decoded
                    let kind = self.take_memory_fault(true);
                    self.state.set_pc(pc);
                    return Err(kind);
                }
                match translated.map_err(ExceptionKind::UnsupportedInstruction)? {
                    Translation::SingleStep => {
                        self.blocks[index] = Some(BlockSlot::single_step(tag));
                        return Ok(Loaded::INTERPRETED);
                    }
                    Translation::Block(block) => {
                        // An exhausted quota means "not compiled this turn", not "this address cannot
                        // be compiled": the former keeps `denied` so the next turn retries, the
                        // latter must not retry (it would waste a decode every turn).
                        let mut denied = false;
                        let compiled = if self.jit_enabled {
                            if self.take_compile_budget() {
                                if self.jit_arena.len() >= self.jit_arena_limit {
                                    // Invalidate before compiling: the new block must not be pushed
                                    // into the arena and then released along with it, or the returned
                                    // entry would point into executable memory that has been freed.
                                    // Invalidation also sets every link slot back to "unlinked", so
                                    // no jump target dangles.
                                    self.invalidate_code_cache();
                                }
                                // The flat windows are the memory implementation's business; without
                                // any, the callback path is used. The pointers must stay valid while
                                // the block lives, which is `Memory::fast_window`'s contract.
                                let windows = if self.fastmem_enabled {
                                    self.memory.fast_windows()
                                } else {
                                    [None; crate::memory::MAX_WINDOWS]
                                };
                                // Running out of code memory also means invalidate-then-compile, for
                                // the same reason as the block ceiling: once the arena is reset, the
                                // entries compiled before it point at positions that will be
                                // overwritten.
                                if self.code_arena.remaining() < crate::backend::MIN_BLOCK_ROOM {
                                    self.invalidate_code_cache();
                                }
                                match backend::compile::<M>(
                                    &block,
                                    thumb,
                                    windows,
                                    &mut self.code_arena,
                                ) {
                                    Some(jit) => self.register_compiled(jit),
                                    None => CompiledEntry::NONE,
                                }
                            } else {
                                denied = true;
                                CompiledEntry::NONE
                            }
                        } else {
                            CompiledEntry::NONE
                        };
                        self.blocks[index] = Some(BlockSlot {
                            tag,
                            entry: compiled.entry,
                            count: block.instruction_count,
                            next_pc: block.next_pc,
                            body: compiled.body,
                            frame: compiled.frame,
                            exchange: compiled.exchange,
                            block: Some(Box::new(block)),
                            denied,
                        });
                    }
                }
            }

            match &self.blocks[index] {
                Some(slot) => Ok(Loaded {
                    entry: slot.entry,
                    body: slot.body,
                    frame: slot.frame,
                    exchange: slot.exchange,
                }),
                None => unreachable!("the cache slot was just filled"),
            }
        }
    }

    /// Interprets one block (or a single step).
    fn run_interpreted(
        &mut self,
        pc: u32,
        thumb: bool,
        executed: &mut u64,
    ) -> Result<(), ExceptionKind> {
        self.state.materialize_lazy_flags();
        let index = cache_index(pc, thumb);
        let tag = cache_tag(pc, thumb);
        debug_assert!(matches!(&self.blocks[index], Some(slot) if slot.tag == tag));

        let flow = {
            let slot = self.blocks[index]
                .as_ref()
                .expect("the cache slot was just filled");
            let Some(block) = slot.block.as_ref() else {
                // Known not to form a block: single-step
                self.step()?;
                *executed += 1;
                return Ok(());
            };
            let state = &mut self.state;
            let memory = &mut self.memory;
            let mut executor = Executor::new(state, memory);
            if block.has_cond {
                executor.execute_guarded(&block.ops)
            } else {
                executor.execute(&block.ops)
            }
        };

        let slot = self.blocks[index]
            .as_ref()
            .expect("the cache slot was just filled");
        *executed += slot.count as u64;
        let (next_pc, terminal_pc) = (slot.next_pc, slot.block.as_ref().map(|b| b.terminal_pc));

        match flow {
            Flow::Next => self.state.set_pc(next_pc),
            Flow::Branch(target) => self.state.set_pc(target),
            Flow::Exchange(target) => {
                self.state.cpsr.set_t(target & 1 != 0);
                self.state.set_pc(target & !1);
            }
            Flow::Exception(kind) => {
                if let Some(terminal_pc) = terminal_pc {
                    self.state.set_pc(terminal_pc);
                }
                return Err(kind);
            }
        }
        Ok(())
    }

    /// Runs continuously, returning how many instructions actually executed.
    pub fn run(&mut self, max_instructions: u64) -> Result<u64, ExceptionKind> {
        let mut count = 0;
        while count < max_instructions {
            self.step()?;
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SimpleMemory;

    #[test]
    fn jit_arena_limit_invalidates_before_compiling() {
        let mut memory = SimpleMemory::new(0, 0x100);
        memory.write_words(
            0,
            &[
                0xEAFF_FFFF, // 0: b 4
                0xEAFF_FFFF, // 4: b 8
                0xEAFF_FFFC, // 8: b 0
            ],
        );
        let mut cpu = Cpu::new(memory);
        cpu.jit_arena_limit = 2;
        cpu.state.set_pc(0);

        // Three blocks run in a cycle, so a limit of 2 forces repeated invalidation; the entries
        // must come from the arena throughout
        cpu.run_cached(12).unwrap();

        assert!(cpu.jit_arena.len() <= 2);
        assert_eq!(cpu.state.pc(), 0);
        assert!(cpu.compiled_block_count() > 0);
    }

    /// Hot blocks from a real workload that evict each other: their addresses are `0x14000` and
    /// `0x42000` apart, exactly the strides low-bit direct mapping collides on. Under direct mapping
    /// the two hot blocks evict each other every round and each recompiles, which makes the JIT
    /// slower than the interpreter. The index has to take the high bits into account.
    #[test]
    fn cache_index_does_not_alias_power_of_two_strided_blocks() {
        for (a, b) in [
            (0x1003_9388u32, 0x1002_5388u32),
            (0x1000_5e10, 0x1004_7e10),
            (0x1000_5e2c, 0x1004_7e2c),
            (0x1003_93ac, 0x1002_53ac),
            (0x1003_53ac, 0x1001_53ac),
        ] {
            assert_ne!(
                cache_index(a, false),
                cache_index(b, false),
                "{a:#x} and {b:#x} sharing a slot would evict each other and recompile every round"
            );
        }
        assert_ne!(
            cache_index(0x1000_0000, false),
            cache_index(0x1004_0000, false),
            "the high bits have to take part in the index"
        );
        assert_ne!(
            cache_index(0x1000_0000, false),
            cache_index(0x1000_0000, true),
            "the ARM and Thumb blocks at one address must not share a slot either"
        );
    }
}
