//! Basic-block translation and caching.
//!
//! A basic block is translated from `start_pc` onwards until a terminal instruction appears, until
//! a conditional instruction whose condition cannot be settled statically, or until the instruction
//! cap is reached. The result can be cached and run repeatedly, which is the first payoff of block
//! translation over decoding one instruction at a time.
//!
//! **Conditional execution**: a conditional branch becomes `BranchConditional` and a non-branch
//! conditional instruction becomes a `Cond` guard. Both conditions are evaluated at **run time**,
//! which is what makes a block safe to cache.
//!
//! **PC as an operand**: r15 does not advance inside a block. PC-relative addressing is folded into
//! absolute addresses at translation time; every other shape that reads r15 (`BX pc`,
//! `mul ..., pc`, a register-specified shift amount taken from pc, and so on) is detected by
//! [`crate::ir::facts::reads_pc_before_write`] and falls back to single-stepping.
//!
//! **Instruction-set switches**: `MSR CPSR_c` can change the T bit, so the translator ends the
//! block after that instruction rather than decoding what follows under the old instruction set.

use crate::decoder::{DecodeError, a32, thumb};
use crate::ir::facts::reads_pc_before_write;
use crate::ir::{Address, AluOp, IrOp, ShiftType, Shifter, Value};
use crate::memory::Memory;
use crate::state::Cpsr;

/// The most guest instructions one block may hold, so that a single translation does not grow long
/// enough to hurt the hit rate.
pub const MAX_INSTRUCTIONS_PER_BLOCK: u32 = 32;

/// What a block's entry expects of the deferred flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryFlags {
    /// The block neither reads nor writes the condition flags.
    NoAccess,
    /// The first access is an arithmetic instruction that writes all of NZCV, so the old flags can
    /// simply be discarded.
    Discards,
    /// The first access is a read, or a partial write, so the old flags must be materialized
    /// first.
    NeedsMaterialized,
}

/// A translated basic block.
#[derive(Debug)]
pub struct BasicBlock {
    /// The block's first address.
    pub start_pc: u32,
    /// Where execution continues when no jump is taken.
    pub next_pc: u32,
    /// The address of the last instruction; on an exception PC stops here.
    pub terminal_pc: u32,
    /// How many guest instructions the block holds (used for execution counting and budgeting).
    pub instruction_count: u32,
    /// The expanded IR sequence.
    pub ops: Vec<IrOp>,
    /// Whether the block contains condition guards ([`IrOp::Cond`]).
    ///
    /// Kept as a flag so the executor can run two separate loops: the common guard-free block takes
    /// the tightest possible `for` iteration, sparing a test per operation.
    pub has_cond: bool,
    /// What the entry requires of the deferred flags.
    pub entry_flags: EntryFlags,
    /// Whether the condition flags are still live at each IR position: `flag_live_at[p]` being
    /// false means that from position `p` onwards, up to the next full write, **nothing reads
    /// them**. (`p == ops.len()` is the block exit and is always true: the flags at the exit may be
    /// live for a successor.)
    ///
    /// The result of a **must** analysis, used only to take things away: where it is false the
    /// backend can stop maintaining the flags (the commonest case being "the flags are dead past
    /// the merge point" inside a conditional-execution block). Where it is true the flags are
    /// treated as live, so a conservative analysis only forgoes an optimization and never computes
    /// the wrong answer.
    pub flag_live_at: Vec<bool>,
}

impl BasicBlock {
    /// Whether the flags are already dead after position `pos`, i.e. nothing reads them.
    pub fn flags_dead_at(&self, pos: usize) -> bool {
        !self.flag_live_at.get(pos).copied().unwrap_or(true)
    }
}

/// The outcome of a translation.
#[derive(Debug)]
pub enum Translation {
    /// A cacheable block.
    Block(BasicBlock),
    /// This address cannot form a block (a conditional data instruction, a Thumb `BL` prefix, and
    /// so on); the caller should fall back to single-stepping.
    SingleStep,
}

/// Translates one basic block starting at `start_pc`.
pub fn translate<M: Memory>(
    memory: &mut M,
    start_pc: u32,
    is_thumb: bool,
) -> Result<Translation, DecodeError> {
    let mut ops: Vec<IrOp> = Vec::with_capacity(16);
    let mut pc = start_pc;
    let mut count: u32 = 0;
    // The address of the last instruction **taken into this block**; on an exception PC stops here
    let mut terminal_pc = start_pc;

    while count < MAX_INSTRUCTIONS_PER_BLOCK {
        let instruction_pc = pc;
        let result = if is_thumb {
            translate_thumb_instruction(memory, instruction_pc, &mut ops)
        } else {
            translate_arm_instruction(memory, instruction_pc, &mut ops)
        };

        let (length, action) = match result {
            Ok(translated) => translated,
            Err(err) => {
                // An illegal or unsupported instruction inside the block: end the block first so
                // that the instructions before it still run, and fail again when a translation
                // starts here and report the error to the caller. That matches the single-step
                // path, which also executes what came before and then reports.
                if count == 0 {
                    return Err(err);
                }
                break;
            }
        };

        if matches!(action, Action::SingleStep) {
            // This instruction demands single-stepping (a conditional data instruction, a PC-based
            // form, and so on), so the block ends here
            if count == 0 {
                return Ok(Translation::SingleStep);
            }
            break;
        }

        terminal_pc = instruction_pc;
        count += 1;
        pc = instruction_pc.wrapping_add(length);

        if matches!(action, Action::EndBlock) {
            break;
        }
    }

    // Block-level optimization: copy/constant propagation plus dead-register-write elimination,
    // then dead-flag elimination. The order matters: value propagation can turn a flag write into a
    // dead one, so running it first gives flag elimination its full opportunity.
    crate::ir::optimize(&mut ops, start_pc);
    let flag_live_at = eliminate_dead_flags(&mut ops);

    let entry_flags = classify_entry_flags(&ops);
    Ok(Translation::Block(BasicBlock {
        start_pc,
        next_pc: pc,
        terminal_pc,
        instruction_count: count,
        has_cond: ops.iter().any(|op| matches!(op, IrOp::Cond { .. })),
        flag_live_at,
        ops,
        entry_flags,
    }))
}

enum Action {
    Continue,
    EndBlock,
    SingleStep,
}

/// Translates one A32 instruction, rolling back every IR operation it left behind if decoding fails.
///
/// The decoder may only discover an illegal encoding **after** it has already pushed operations (a
/// `data_processing` form emits `Alu` first and then rejects the exception-return shape with S=1
/// writing PC, say). Without the rollback that rejected instruction would sit at the block tail as
/// an ordinary operation and really execute once, while the single-step path does not execute it —
/// and the two paths would then disagree about where the failure happened.
fn translate_arm_instruction<M: Memory>(
    memory: &mut M,
    pc: u32,
    ops: &mut Vec<IrOp>,
) -> Result<(u32, Action), DecodeError> {
    let start = ops.len();
    let result = translate_arm_instruction_raw(memory, pc, ops);
    if result.is_err() {
        ops.truncate(start);
    }
    result
}

fn translate_arm_instruction_raw<M: Memory>(
    memory: &mut M,
    pc: u32,
    ops: &mut Vec<IrOp>,
) -> Result<(u32, Action), DecodeError> {
    let inst = memory.read_u32(pc);
    let cond = inst >> 28;

    // AL and the unconditional space: translate directly, with no condition to test in machine code
    if cond == 0xE || cond == 0xF {
        let before = ops.len();
        a32::decode_into(inst, pc, true, ops)?;
        // r15 does not advance inside a block, so an instruction that uses PC as data has to fall
        // back to single-stepping
        if reads_pc_before_write(&ops[before..]) {
            ops.truncate(before);
            return Ok((4, Action::SingleStep));
        }
        let terminal = ops.len() > before && last_is_terminal(ops);
        let switches_isa = ops[before..].iter().any(writes_instruction_set);
        return Ok((
            4,
            if terminal || switches_isa {
                Action::EndBlock
            } else {
                Action::Continue
            },
        ));
    }

    // A conditional branch: the condition is left to run time, so the block stays cacheable.
    //
    // The block ends here. `BranchConditional` can already decide at run time whether to jump, so
    // in principle translation could carry on and fold the fallthrough into the same block —
    // **measured, it is not worth it**: over a 342 KiB code segment of a real application (translate
    // only, never execute) the mean block length rose merely from 8.52 to 10.35 instructions
    // (+21%), a fall of about 17% in block count, because short blocks are caused by the
    // **unconditional** terminators (`b`, `bl`, `pop pc`), not by conditional branches. Trading
    // that 17% for disturbing the early-exit instruction accounting inside a block (`slot.count`
    // would read high and `run_cached(n)`'s budget would stop being accurate) is a far worse
    // bargain.
    if a32::is_branch(inst) {
        a32::decode_branch_conditional(inst, pc, cond, ops);
        return Ok((4, Action::EndBlock));
    }

    // Conditional data processing, access, or multiply: emit a condition guard first, then the
    // instruction's operations. The guard only skips what immediately follows, and the condition is
    // still evaluated at run time from the flags as they stand, so the block stays safe to cache.
    let guard_index = ops.len();
    ops.push(IrOp::Cond { cond, skip: 0 });
    let before = ops.len();
    a32::decode_into(inst, pc, true, ops)?;

    if reads_pc_before_write(&ops[before..]) {
        ops.truncate(guard_index);
        return Ok((4, Action::SingleStep));
    }

    let skip = ops.len() - before;
    if skip == 0 {
        // Nothing executable came out of the expansion (PLD, say), so the guard is pointless
        ops.truncate(guard_index);
        return Ok((4, Action::Continue));
    }
    ops[guard_index] = IrOp::Cond {
        cond,
        skip: skip as u8,
    };

    let terminal = last_is_terminal(ops);
    let switches_isa = ops[before..].iter().any(writes_instruction_set);
    Ok((
        4,
        if terminal || switches_isa {
            Action::EndBlock
        } else {
            Action::Continue
        },
    ))
}

/// Translates one Thumb instruction, rolling back its IR on failure in the same way as the A32
/// version (see there for why).
fn translate_thumb_instruction<M: Memory>(
    memory: &mut M,
    pc: u32,
    ops: &mut Vec<IrOp>,
) -> Result<(u32, Action), DecodeError> {
    let start = ops.len();
    let result = translate_thumb_instruction_raw(memory, pc, ops);
    if result.is_err() {
        ops.truncate(start);
    }
    result
}

fn translate_thumb_instruction_raw<M: Memory>(
    memory: &mut M,
    pc: u32,
    ops: &mut Vec<IrOp>,
) -> Result<(u32, Action), DecodeError> {
    let inst = memory.read_u16(pc);

    // BL / BLX: translate both halfwords together
    if thumb::is_bl_prefix(inst) {
        let lower = memory.read_u16(pc.wrapping_add(2));
        let pair = thumb::decode_bl_pair(inst, lower, pc)?;
        ops.extend(pair);
        return Ok((4, Action::EndBlock));
    }

    // B<cond>: the condition is settled at run time (same reasoning as A32: folding the fallthrough
    // in measured out at only 17%, so it is not done)
    if thumb::is_conditional_branch(inst) {
        thumb::decode_conditional_branch(inst, pc, ops);
        return Ok((2, Action::EndBlock));
    }

    let before = ops.len();
    thumb::decode_into(inst, pc, true, ops)?;
    // As above: PC-relative loads (`ldr r0, [pc, #x]`) are very common in Thumb code and cannot go
    // into a block, or they would read a stale r15
    if reads_pc_before_write(&ops[before..]) {
        ops.truncate(before);
        return Ok((2, Action::SingleStep));
    }
    let terminal = ops.len() > before && last_is_terminal(ops);
    Ok((
        2,
        if terminal {
            Action::EndBlock
        } else {
            Action::Continue
        },
    ))
}

fn last_is_terminal(ops: &[IrOp]) -> bool {
    ops.last().is_some_and(|op| op.is_terminal())
}

/// Whether the operation may switch the current instruction set.
///
/// `MSR CPSR_c` can change the T bit, and the instructions after it in the block were already
/// decoded under the old instruction set, so the block has to end here.
fn writes_instruction_set(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::WriteStatus {
            mask,
            spsr: false,
            ..
        } if *mask & Cpsr::T != 0
    )
}

// ---------------------------------------------------------------------------
// Dead-flag elimination
// ---------------------------------------------------------------------------

const F_N: u8 = 1;
const F_Z: u8 = 2;
const F_C: u8 = 4;
const F_V: u8 = 8;

/// Whether the operand **defines** the shifter carry.
///
/// The crucial distinction: with a shift amount of zero (or an immediate rotate of zero) the C flag
/// **keeps its value** rather than being written. Treating such an instruction as "writes C" would
/// judge an earlier C write dead — and the `BCS` after a `MOVS r0, r1` would read the wrong carry.
fn value_defines_carry(value: Value) -> bool {
    match value {
        Value::ImmC { .. } => true,
        Value::Shifted {
            shift,
            by: Shifter::Imm(amount),
            ..
        } => shift == ShiftType::Rrx || amount != 0,
        // A register-specified shift amount may be zero at run time, in which case C keeps its
        // value; when it is non-zero, C is written. Liveness treats it as "may write C" and also
        // marks it as reading C in flag_read_mask, so neither an earlier C write is deleted by
        // mistake nor the C it produces itself goes missing.
        Value::Shifted {
            by: Shifter::Reg(_),
            ..
        } => true,
        _ => false,
    }
}

/// Which flag bits a condition code depends on.
fn cond_flag_mask(cond: u32) -> u8 {
    match cond {
        0x0 | 0x1 => F_Z,
        0x2 | 0x3 => F_C,
        0x4 | 0x5 => F_N,
        0x6 | 0x7 => F_V,
        0x8 | 0x9 => F_C | F_Z,
        0xA | 0xB => F_N | F_V,
        0xC | 0xD => F_Z | F_N | F_V,
        _ => 0,
    }
}

/// Which condition flags one IR operation writes.
///
/// A guard writes no flags itself, and whether the operations it guards run at all depends on the
/// flags as they stand — so [`eliminate_dead_flags`] handles those separately as "may not execute",
/// and they take no part in the write test here.
fn flag_write_mask(op: &IrOp) -> u8 {
    match op {
        IrOp::Alu {
            op: alu,
            a,
            b,
            set_flags: true,
            ..
        } => {
            if alu.is_logical() {
                let operand = if alu.is_move() { *a } else { b.unwrap_or(*a) };
                let mut mask = F_N | F_Z;
                if value_defines_carry(operand) {
                    mask |= F_C;
                }
                mask
            } else {
                F_N | F_Z | F_C | F_V
            }
        }
        IrOp::Mul {
            set_flags: true, ..
        }
        | IrOp::MulLong {
            set_flags: true, ..
        }
        | IrOp::MulHalf {
            set_flags: true, ..
        }
        | IrOp::MulWordHalf {
            set_flags: true, ..
        } => F_N | F_Z,
        _ => 0,
    }
}

/// Whether this is an arithmetic instruction that writes all of NZCV first and reads no C.
fn lazy_eligible_write(op: &IrOp) -> bool {
    match op {
        IrOp::Alu {
            op,
            a,
            b,
            set_flags: true,
            ..
        } => {
            matches!(
                op,
                AluOp::Add | AluOp::Sub | AluOp::Rsb | AluOp::Cmp | AluOp::Cmn
            ) && !value_reads_carry(*a)
                && !(*b).is_some_and(value_reads_carry)
        }
        _ => false,
    }
}

/// Decides from the block's first flag access whether the old flags have to be materialized on
/// entry.
fn classify_entry_flags(ops: &[IrOp]) -> EntryFlags {
    for op in ops {
        if flag_read_mask(op) != 0 {
            return EntryFlags::NeedsMaterialized;
        }
        // MSR may write the CPSR condition flags; materialize first so that old flags are not
        // merged into the new value
        if matches!(op, IrOp::WriteStatus { spsr: false, .. }) {
            return EntryFlags::NeedsMaterialized;
        }
        if flag_write_mask(op) != 0 {
            return if lazy_eligible_write(op) {
                EntryFlags::Discards
            } else {
                EntryFlags::NeedsMaterialized
            };
        }
    }
    EntryFlags::NoAccess
}

/// Which condition flags one IR operation reads (counting only the reads the **execution semantics**
/// demand).
///
/// The backend uses it too, to tell whether an operation turns the flag representation into a live
/// CPSR, so the two must agree.
pub(crate) fn flag_read_mask(op: &IrOp) -> u8 {
    match op {
        IrOp::BranchConditional { cond, .. } | IrOp::Cond { cond, .. } => cond_flag_mask(*cond),
        // Add and subtract with carry read C directly
        IrOp::Alu {
            op: AluOp::Adc | AluOp::Sbc | AluOp::Rsc,
            ..
        } => F_C,
        // RRX's result depends on C
        IrOp::Alu { a, b, .. } if value_reads_carry(*a) || (*b).is_some_and(value_reads_carry) => {
            F_C
        }
        // A logical instruction's register-specified shift amount may be zero, in which case C is
        // left as it was and an earlier definition is still live
        IrOp::Alu {
            op,
            a,
            b,
            set_flags: true,
            ..
        } if (*op).is_logical() && uses_register_shift(*op, *a, *b) => F_C,
        // An RRX offset in an access address reads C as well
        IrOp::Load { address, .. }
        | IrOp::Store { address, .. }
        | IrOp::LoadDouble { address, .. }
        | IrOp::StoreDouble { address, .. }
        | IrOp::Swap { address, .. }
            if address_reads_carry(address) =>
        {
            F_C
        }
        // MRS reads the whole CPSR out
        IrOp::ReadStatus { .. } => F_N | F_Z | F_C | F_V,
        _ => 0,
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

/// Whether an access address's offset uses RRX.
fn address_reads_carry(address: &Address) -> bool {
    match address {
        Address::Absolute(_) => false,
        Address::Register { offset, .. } => value_reads_carry(*offset),
    }
}

/// Whether the instruction's effective second operand uses a register-specified shift amount.
fn uses_register_shift(op: AluOp, a: Value, b: Option<Value>) -> bool {
    let operand = if op.is_move() { a } else { b.unwrap_or(a) };
    matches!(
        operand,
        Value::Shifted {
            by: Shifter::Reg(_),
            ..
        }
    )
}

/// Eliminates dead flag writes inside the block, returning the flag-liveness table aligned with the
/// IR (see [`BasicBlock::flag_live_at`]).
///
/// Liveness runs backwards from the block's end: a flag write with no read **before it is
/// overwritten** is dead. Note that the flags at a block exit may be live for a successor, so the
/// initial liveness takes all four bits (conservative, but it keeps the result correct across
/// blocks; CFG-based cross-block analysis is left for later).
fn eliminate_dead_flags(ops: &mut Vec<IrOp>) -> Vec<bool> {
    let mut live = F_N | F_Z | F_C | F_V;
    let mut dead = vec![false; ops.len()];
    let mut live_at = vec![true; ops.len() + 1];

    // Mark the operations covered by a condition guard.
    //
    // Those operations **may not run at all**, so their flag writes cannot be treated as "killing"
    // an earlier write (the earlier carry would be lost when the condition fails), and they cannot
    // be deleted either (deleting one would throw off the `skip` count already computed for the
    // `Cond`).
    let mut conditional = vec![false; ops.len()];
    let mut cursor = 0;
    while cursor < ops.len() {
        if let IrOp::Cond { skip, .. } = ops[cursor] {
            let end = (cursor + 1 + skip as usize).min(ops.len());
            for slot in &mut conditional[cursor + 1..end] {
                *slot = true;
            }
            cursor = end;
        } else {
            cursor += 1;
        }
    }

    // Backward data flow: entry liveness = (exit liveness & !write) | read.
    //
    // Note that `| read` has to come **after** `& !write`: an instruction like `ADCS` both reads
    // and writes the same flag, and the C it reads comes from an **earlier** definition. Folding
    // the read in first and then clearing everything would judge the very C write ADCS depends on
    // dead.
    for (index, op) in ops.iter().enumerate().rev() {
        let read = flag_read_mask(op);
        let write = flag_write_mask(op);

        // A guarded write can be dead as well: if it runs it writes only these bits, if it does not
        // run nothing happens, and either way nobody reads them — so the flag side effect alone can
        // go (the operation stays, because `skip` is a fixed number). But it **still cannot** count
        // as killing an earlier write: with the condition false, the bits written earlier are still
        // live.
        if write != 0 && write & live == 0 {
            dead[index] = true;
        }
        if conditional[index] {
            live |= read;
        } else {
            live = (live & !write) | read;
        }
        live_at[index] = live != 0;
    }

    if !dead.iter().any(|is_dead| *is_dead) {
        return live_at;
    }

    let tail = live_at[ops.len()];
    let mut rebuilt = Vec::with_capacity(ops.len());
    let mut rebuilt_live = Vec::with_capacity(ops.len() + 1);
    for (index, op) in ops.drain(..).enumerate() {
        if !dead[index] {
            rebuilt.push(op);
            // The liveness table must line up with the final ops **item for item**: a dropped
            // operation may not keep its slot, or the backend reads the liveness of the wrong
            // position when deciding whether the flags are dead. A differential test caught this
            // once.
            rebuilt_live.push(live_at[index]);
            continue;
        }
        // A guarded operation cannot be deleted (`Cond`'s skip is a fixed number computed at
        // translation time), so it can only degrade to an operation that does nothing.
        let keep_as_nop = conditional[index];
        let before = rebuilt.len();
        // Keep an instruction that writes a result and drop only its flag side effect; a pure test
        // instruction (CMP/TST and the like) goes entirely
        match op {
            IrOp::Alu {
                op,
                dst,
                a,
                b,
                set_flags: _,
            } if dst.is_some() => rebuilt.push(IrOp::Alu {
                op,
                dst,
                a,
                b,
                set_flags: false,
            }),
            IrOp::Alu { .. } if keep_as_nop => rebuilt.push(IrOp::Nop),
            IrOp::Mul {
                dst,
                a,
                b,
                accumulate,
                set_flags: _,
            } => rebuilt.push(IrOp::Mul {
                dst,
                a,
                b,
                accumulate,
                set_flags: false,
            }),
            IrOp::MulLong {
                lo,
                hi,
                a,
                b,
                signed,
                accumulate,
                set_flags: _,
            } => rebuilt.push(IrOp::MulLong {
                lo,
                hi,
                a,
                b,
                signed,
                accumulate,
                set_flags: false,
            }),
            IrOp::MulHalf {
                dst,
                addend,
                a,
                b,
                a_high,
                b_high,
                long,
                set_flags: _,
            } => rebuilt.push(IrOp::MulHalf {
                dst,
                addend,
                a,
                b,
                a_high,
                b_high,
                long,
                set_flags: false,
            }),
            IrOp::MulWordHalf {
                dst,
                addend,
                a,
                b,
                b_high,
                set_flags: _,
            } => rebuilt.push(IrOp::MulWordHalf {
                dst,
                addend,
                a,
                b,
                b_high,
                set_flags: false,
            }),
            // A pure test instruction: no observable effect at all
            _ => {
                if keep_as_nop {
                    rebuilt.push(IrOp::Nop);
                }
            }
        }
        if rebuilt.len() > before {
            rebuilt_live.push(live_at[index]);
        }
    }
    rebuilt_live.push(tail);
    *ops = rebuilt;
    rebuilt_live
}
