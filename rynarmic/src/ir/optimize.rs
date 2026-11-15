//! Block-local copy and constant propagation plus dead-register-write elimination.
//!
//! Runs once after [`crate::block::translate`] has produced the IR and before a backend compiles it.
//!
//! For the switches see [`optimize`]: everything is on by default, and `RYNARMIC_IR_OPT=off|prop|dce|full`
//! steps it back one stage at a time (`RYNARMIC_IR_DEBUG=1` dumps the IR before and after each
//! block that changed).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::facts::{MAX_WRITES, reads_reg, writes_regs};
use super::{Address, AluOp, IrOp, Shifter, Value};

/// How many blocks were optimized (used by `totals`).
static BLOCKS: AtomicUsize = AtomicUsize::new(0);
static OPS_BEFORE: AtomicUsize = AtomicUsize::new(0);
static OPS_AFTER: AtomicUsize = AtomicUsize::new(0);

/// Cumulative `(blocks, IR operations before, IR operations after)`.
///
/// For benchmarks and reports only (`compiled_bench` prints it); no semantics depend on it.
pub fn totals() -> (usize, usize, usize) {
    (
        BLOCKS.load(Ordering::Relaxed),
        OPS_BEFORE.load(Ordering::Relaxed),
        OPS_AFTER.load(Ordering::Relaxed),
    )
}

/// `RYNARMIC_IR_DEBUG` is read once, on the first enquiry.
///
/// This test sits on **every block translation**, and `std::env::var_os` takes a process-wide lock
/// and walks the environment block (microseconds at a time on Windows) — left on the hot path it
/// drags one block translation from about 200 ns to 19 µs and destroys the "cold-start a real
/// module" numbers outright. An environment variable is not supposed to change mid-run anyway.
fn debug_on() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("RYNARMIC_IR_DEBUG").is_some())
}

/// The optimization level. `RYNARMIC_IR_OPT=off|prop|dce|full`, everything on when unset.
///
/// The level is read once per process: `translate` calls into this module on every miss, so reading
/// the environment block by block is both slow (`env::var` allocates and takes a lock) and makes
/// behaviour drift with the environment inside one process.
fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("RYNARMIC_IR_OPT").as_deref() {
        Ok("off") => Mode::Off,
        Ok("prop") => Mode::Prop,
        Ok("dce") => Mode::Dce,
        _ => Mode::Full,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Prop,
    Dce,
    Full,
}

/// What is known about a register: the equivalence an earlier operation in the block established.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Known {
    /// Nothing.
    Unknown,
    /// Equivalent to another register (copy propagation).
    Reg(u8),
    /// Equivalent to a constant (constant propagation).
    Const(u32),
}

/// Rewrites one operand from the known-values table (copy and constant propagation).
///
/// The **shift-amount register** of a `Value::Shifted` is rewritten too: the shift amount's value is
/// tracked by the same table, and swapping which register holds it changes nothing.
fn propagate(value: &mut Value, known: &[Known; 16]) {
    match value {
        Value::Reg(r) => match known[*r as usize] {
            Known::Const(c) => *value = Value::Imm(c),
            Known::Reg(s) if s != *r => *value = Value::Reg(s),
            _ => {}
        },
        Value::Shifted { reg, by, .. } => {
            if let Known::Reg(s) = known[*reg as usize]
                && s != *reg
            {
                *reg = s;
            }
            if let Shifter::Reg(s) = by
                && let Known::Reg(t) = known[*s as usize]
                && t != *s
            {
                *s = t;
            }
        }
        _ => {}
    }
}

/// Invalidates what is known about `reg`, along with every copy entry pointing at it.
///
/// After `mov r1, r0` the table holds `known[r1] = Reg(r0)`: that knowledge is about r0's **old**
/// value, so once r0 is overwritten it becomes a dangling reference and has to be dropped with it.
fn invalidate(known: &mut [Known; 16], reg: u8) {
    known[reg as usize] = Known::Unknown;
    for entry in known.iter_mut() {
        if *entry == Known::Reg(reg) {
            *entry = Known::Unknown;
        }
    }
}

/// Rewrites one register number from the known-values table (copy propagation; a constant cannot be
/// written as a register number, so it is skipped).
fn propagate_reg(reg: &mut u8, known: &[Known; 16]) {
    if let Known::Reg(s) = known[*reg as usize]
        && s != *reg
    {
        *reg = s;
    }
}

/// Block-local optimization: copy and constant propagation plus local dead-write elimination.
///
/// The safety boundaries, none of which may be dropped:
/// - operations inside a `Cond` guard group are **skipped as a group**: a write there may not
///   execute (so it cannot enter the known-values table) and the `skip` count is fixed at
///   translation time (so neither an operation may be deleted nor the count changed);
/// - an `Alu` with `set_flags` is not rewritten: a logical operation's C comes from the shifter, and
///   turning `Reg` into `Imm` would change where C comes from;
/// - dead-write elimination only removes pure register writes (accesses, writebacks and
///   multi-register moves all stay), and every register is live at the block exit by default,
///   because registers are global state and what is live outside the block is unknown.
pub fn optimize(ops: &mut Vec<IrOp>, start_pc: u32) {
    let (do_prop, do_dce) = match mode() {
        Mode::Off => return,
        Mode::Prop => (true, false),
        Mode::Dce => (false, true),
        Mode::Full => (true, true),
    };
    let before_len = ops.len();
    if debug_on() {
        // Triage: dump everything before and after a change (block start pc included), to locate
        // the deletion logged inside optimize_inner
        let before: Vec<String> = ops.iter().map(|op| format!("  {op:?}")).collect();
        optimize_inner(ops, do_prop, do_dce);
        let after: Vec<String> = ops.iter().map(|op| format!("  {op:?}")).collect();
        if before != after {
            eprintln!(
                "=== optimize pc={start_pc:#x} ===\nbefore:\n{}\nafter:\n{}",
                before.join("\n"),
                after.join("\n")
            );
        }
    } else {
        optimize_inner(ops, do_prop, do_dce);
    }
    BLOCKS.fetch_add(1, Ordering::Relaxed);
    OPS_BEFORE.fetch_add(before_len, Ordering::Relaxed);
    OPS_AFTER.fetch_add(ops.len(), Ordering::Relaxed);
}

/// Rewrites one operation's operands from the known-values table.
///
/// Only "which register / which constant" changes, never the operation's kind and never the number
/// of operands — the latter would break the counting semantics of `Cond { skip }`.
fn rewrite_operands(op: &mut IrOp, known: &[Known; 16]) {
    match op {
        IrOp::Alu {
            a, b, set_flags, ..
        } => {
            if !*set_flags {
                propagate(a, known);
                if let Some(b) = b {
                    propagate(b, known);
                }
            }
        }
        IrOp::Mul {
            a, b, accumulate, ..
        } => {
            propagate_reg(a, known);
            propagate_reg(b, known);
            if let Some(r) = accumulate {
                propagate_reg(r, known);
            }
        }
        IrOp::MulHalf { a, b, addend, .. } | IrOp::MulWordHalf { a, b, addend, .. } => {
            propagate_reg(a, known);
            propagate_reg(b, known);
            if let Some(r) = addend {
                propagate_reg(r, known);
            }
        }
        IrOp::Saturating { a, b, .. } => {
            propagate_reg(a, known);
            propagate_reg(b, known);
        }
        IrOp::Clz { src, .. } => propagate_reg(src, known),
        IrOp::Swap { src, address, .. } => {
            propagate_reg(src, known);
            if let Address::Register { offset, .. } = address {
                propagate(offset, known);
            }
        }
        IrOp::Load { address, .. } | IrOp::LoadDouble { address, .. } => {
            if let Address::Register { offset, .. } = address {
                propagate(offset, known);
            }
        }
        IrOp::Store { src, address, .. } => {
            propagate_reg(src, known);
            if let Address::Register { offset, .. } = address {
                propagate(offset, known);
            }
        }
        IrOp::StoreDouble {
            address: Address::Register { offset, .. },
            ..
        } => propagate(offset, known),
        IrOp::WriteStatus { src, .. } => propagate(src, known),
        IrOp::Jump { target } | IrOp::BranchConditional { target, .. } => propagate(target, known),
        IrOp::JumpExchange { target } => propagate(target, known),
        // Nothing else has a register operand to rewrite: the lists of `LoadMultiple` and
        // `StoreMultiple` are bitmasks that cannot be turned into an "equivalent register", and
        // `Cond`/`Nop`/the exception kinds carry no operands at all.
        _ => {}
    }
}

fn optimize_inner(ops: &mut Vec<IrOp>, do_prop: bool, do_dce: bool) {
    // The extent of each Cond guard
    let mut guarded = vec![false; ops.len()];
    let mut cursor = 0;
    while cursor < ops.len() {
        if let IrOp::Cond { skip, .. } = ops[cursor] {
            let end = (cursor + 1 + skip as usize).min(ops.len());
            for slot in &mut guarded[cursor + 1..end] {
                *slot = true;
            }
            cursor = end;
        } else {
            cursor += 1;
        }
    }

    let mut known = [Known::Unknown; 16];

    // ---- Forward: rewrite operands and maintain the known-values table ----
    let mut index = 0;
    while index < ops.len() {
        if do_prop && !guarded[index] {
            rewrite_operands(&mut ops[index], &known);
        }

        // A guard group: a write inside it may not execute, so skip the whole group and, once past
        // it, invalidate every register the group wrote
        if let IrOp::Cond { skip, .. } = ops[index] {
            let end = (index + 1 + skip as usize).min(ops.len());
            let mut dirtied = [false; 16];
            for op in &ops[index + 1..end] {
                let mut ws = [0u8; MAX_WRITES];
                let n = writes_regs(op, &mut ws);
                for r in &ws[..n] {
                    dirtied[*r as usize] = true;
                }
            }
            for (r, dirty) in dirtied.iter().enumerate() {
                if *dirty {
                    invalidate(&mut known, r as u8);
                }
            }
            index = end;
            continue;
        }

        // An unconditional write invalidates what was known. Note that besides the written register
        // itself, every copy entry pointing at it goes too (the `known[r1] = Reg(r0)` established by
        // `mov r1, r0`): those describe r0's **old** value, and r0 is now a dangling reference.
        let mut writes = [0u8; MAX_WRITES];
        let n = writes_regs(&ops[index], &mut writes);
        for r in &writes[..n] {
            invalidate(&mut known, *r);
        }
        if let IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(dst),
            a,
            set_flags: _,
            b: None,
        } = &ops[index]
            && !guarded[index]
        {
            known[*dst as usize] = match a {
                Value::Reg(s) => known[*s as usize],
                Value::Imm(c) => Known::Const(*c),
                _ => Known::Unknown,
            };
        }
        index += 1;
    }

    // ---- Backward: dead-register-write elimination (nothing removed inside a group, nothing with
    // a flag side effect) ----
    // Every register is live at the block exit (registers are global state, and what is live outside
    // the block is unknown); a write can only be called dead when a later write in the same block
    // overwrites it with no read in between (store-to-store forwarding). With do_dce = false, live
    // stays true throughout, so no write is ever judged dead.
    let mut live = [true; 16];
    let mut dead = vec![false; ops.len()];
    for (index, op) in ops.iter().enumerate().rev() {
        // `mov rd, rs` with rd == rs is a structural no-op (deletable only outside a group)
        if let IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(dst),
            a: Value::Reg(s),
            b: None,
            set_flags: false,
        } = op
            && *dst == *s
            && !guarded[index]
            && do_dce
        {
            dead[index] = true;
            continue;
        }
        if guarded[index] {
            // A guarded operation may or may not execute: if it does, its inputs have to be read
            // (so reads are live); if it does not, **an earlier write to the same register is still
            // visible** (so writes are live too). Both have to be folded in, or an unconditional
            // write before the guarded write is deleted by mistake — and the value is wrong whenever
            // the condition fails.
            let mut ws = [0u8; MAX_WRITES];
            let n = writes_regs(op, &mut ws);
            for r in 0..16u8 {
                if reads_reg(op, r) || ws[..n].contains(&r) {
                    live[r as usize] = true;
                }
            }
            continue;
        }
        let pure_write = match op {
            IrOp::Alu {
                dst: Some(_),
                set_flags,
                ..
            } => !*set_flags,
            IrOp::Mul { set_flags, .. }
            | IrOp::MulLong { set_flags, .. }
            | IrOp::MulHalf { set_flags, .. }
            | IrOp::MulWordHalf { set_flags, .. } => !*set_flags,
            IrOp::Saturating { .. } | IrOp::Clz { .. } | IrOp::ReadStatus { .. } => true,
            _ => false,
        };
        if pure_write && do_dce {
            let mut writes = [0u8; MAX_WRITES];
            let n = writes_regs(op, &mut writes);
            if n > 0 && writes[..n].iter().all(|r| !live[*r as usize]) {
                if debug_on() {
                    eprintln!("DCE removing [{index}]: {op:?}");
                }
                dead[index] = true;
                // A dead write contributes nothing to liveness, so an earlier write to the same
                // register may be dead too
                continue;
            }
        }
        // L_before = reads ∪ (L_after \ writes).
        // Kill the writes first and only then union in the reads: `Eor r0, r0, imm` both reads and
        // writes r0, and the r0 it reads is precisely what the **previous** instruction wrote.
        // Unioning first and killing afterwould treat that dependency as dead and cascade through
        // the whole Eor chain.
        let mut writes = [0u8; MAX_WRITES];
        let n = writes_regs(op, &mut writes);
        for r in &writes[..n] {
            live[*r as usize] = false;
        }
        for r in 0..16u8 {
            if reads_reg(op, r) {
                live[r as usize] = true;
            }
        }
    }

    if dead.iter().any(|d| *d) {
        let mut rebuilt = Vec::with_capacity(ops.len());
        for (index, op) in ops.drain(..).enumerate() {
            if !dead[index] {
                rebuilt.push(op);
            }
        }
        *ops = rebuilt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::MemSize;

    fn writes_of(op: &IrOp) -> Vec<u8> {
        let mut out = [0u8; MAX_WRITES];
        let n = writes_regs(op, &mut out);
        out[..n].to_vec()
    }

    /// SMLALxy's accumulator is the old value of `dst:dst+1`, which has to count as a read; otherwise
    /// the write that initializes the accumulator is deleted as dead. This once lost a carry.
    #[test]
    fn long_halfword_multiply_reads_the_accumulator() {
        let op = IrOp::MulHalf {
            dst: 12,
            addend: None,
            a: 7,
            b: 10,
            a_high: false,
            b_high: true,
            long: true,
            set_flags: false,
        };
        assert!(
            reads_reg(&op, 12),
            "RdLo carries the accumulator's initial value"
        );
        assert!(
            reads_reg(&op, 13),
            "RdHi carries the accumulator's initial value"
        );
        assert!(!reads_reg(&op, 11));
        assert_eq!(writes_of(&op), vec![12, 13]);
    }

    /// SMULxy (not long) only writes.
    #[test]
    fn short_halfword_multiply_only_writes() {
        let op = IrOp::MulHalf {
            dst: 3,
            addend: None,
            a: 7,
            b: 10,
            a_high: false,
            b_high: false,
            long: false,
            set_flags: false,
        };
        assert!(!reads_reg(&op, 3));
        assert_eq!(writes_of(&op), vec![3]);
    }

    /// MLA's accumulator is a read; UMLAL reads both lo and hi; UMULL reads neither.
    #[test]
    fn multiply_accumulators_are_reads() {
        let mla = IrOp::Mul {
            dst: 1,
            a: 2,
            b: 3,
            accumulate: Some(4),
            set_flags: false,
        };
        assert!(reads_reg(&mla, 4));
        assert_eq!(writes_of(&mla), vec![1]);

        let umlal = IrOp::MulLong {
            lo: 2,
            hi: 3,
            a: 4,
            b: 5,
            signed: false,
            accumulate: true,
            set_flags: false,
        };
        assert!(reads_reg(&umlal, 2) && reads_reg(&umlal, 3));

        let umull = IrOp::MulLong {
            lo: 2,
            hi: 3,
            a: 4,
            b: 5,
            signed: false,
            accumulate: false,
            set_flags: false,
        };
        assert!(!reads_reg(&umull, 2) && !reads_reg(&umull, 3));
    }

    /// One LDM can write fifteen list registers plus the base writeback, and `writes_regs`' buffer
    /// has to hold them all; otherwise it truncates silently and the known-values table keeps stale
    /// entries, which once rewrote a register to the wrong one.
    #[test]
    fn load_multiple_writes_are_not_truncated() {
        let op = IrOp::LoadMultiple {
            base: 0,
            regs: 0b0111_1111_1111_1110, // r1-r14
            add: true,
            pre_index: false,
            writeback: true,
        };
        let w = writes_of(&op);
        assert_eq!(w.len(), 15, "r1-r14 plus the r0 writeback");
        for r in 0..=14u8 {
            assert!(w.contains(&r), "r{r} should be recorded as written");
        }
    }

    /// An access's writeback counts as a write, and the base is also a read.
    #[test]
    fn store_with_writeback_reads_base_and_writes_it() {
        let op = IrOp::Store {
            src: 1,
            address: Address::Register {
                base: 5,
                offset: Value::Imm(8),
                add: true,
            },
            size: MemSize::U32,
            pre_index: true,
            writeback: true,
        };
        assert!(reads_reg(&op, 1) && reads_reg(&op, 5));
        assert!(writes_of(&op).contains(&5));
    }

    /// The known-values table: a copy entry has to be dropped along with the register it copies from.
    #[test]
    fn invalidate_drops_copies_pointing_at_the_register() {
        let mut known = [Known::Unknown; 16];
        known[1] = Known::Reg(0);
        known[2] = Known::Reg(0);
        known[3] = Known::Reg(1);
        invalidate(&mut known, 0);
        assert_eq!(known[1], Known::Unknown);
        assert_eq!(known[2], Known::Unknown);
        assert_eq!(
            known[3],
            Known::Reg(1),
            "r3 points at r1 and has nothing to do with r0"
        );
    }

    /// Operations inside a guard group can neither be deleted nor have their writes entered into the
    /// known-values table.
    #[test]
    fn guarded_ops_are_neither_deleted_nor_propagated_into() {
        let mut ops = vec![
            IrOp::Cond { cond: 0, skip: 1 },
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(1),
                a: Value::Imm(7),
                b: None,
                set_flags: false,
            },
            IrOp::Nop,
        ];
        optimize_inner(&mut ops, true, true);
        assert_eq!(ops.len(), 3, "the guard group has to survive untouched");
        assert!(matches!(ops[0], IrOp::Cond { skip: 1, .. }));
        // After a guarded write, r1's value is unknown: a later read still has to be of r1 itself
        let mut after = vec![
            IrOp::Cond { cond: 0, skip: 1 },
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(1),
                a: Value::Imm(7),
                b: None,
                set_flags: false,
            },
            IrOp::Alu {
                op: AluOp::Add,
                dst: Some(2),
                a: Value::Reg(1),
                b: Some(Value::Imm(1)),
                set_flags: false,
            },
        ];
        optimize_inner(&mut after, true, true);
        assert!(matches!(
            after[2],
            IrOp::Alu {
                a: Value::Reg(1),
                ..
            }
        ));
    }

    /// Every register is live at the block exit, so the last write cannot be deleted as dead.
    #[test]
    fn trailing_write_is_kept() {
        let mut ops = vec![IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(9),
            a: Value::Imm(1),
            b: None,
            set_flags: false,
        }];
        optimize_inner(&mut ops, true, true);
        assert_eq!(ops.len(), 1);
    }

    /// The same register written twice with no read in between: the first write is dead.
    #[test]
    fn overwritten_write_is_dropped() {
        let mut ops = vec![
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(1),
                a: Value::Imm(1),
                b: None,
                set_flags: false,
            },
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(1),
                a: Value::Imm(2),
                b: None,
                set_flags: false,
            },
        ];
        optimize_inner(&mut ops, false, true);
        assert_eq!(
            ops.len(),
            1,
            "the first write is overwritten and nobody reads it"
        );
        assert!(matches!(
            ops[0],
            IrOp::Alu {
                a: Value::Imm(2),
                ..
            }
        ));
    }
}
