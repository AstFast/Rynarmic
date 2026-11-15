//! The **structural facts** of each IR operation: which registers it reads, which it writes, and
//! whether it touches memory.
//!
//! This knowledge used to be written out three times in three modules (the optimizer's dead-store
//! analysis, the block translator's "is PC read part-way through" test, and the JIT's "does this
//! block need memory" test), with the result that **adding one IR operation meant editing four
//! places, and missing one failed silently**. The failures that actually happened:
//!
//! - the dead-store analysis' write buffer had room for four slots, so `LDM` (up to sixteen
//!   writes) was silently truncated; the known-values table kept stale entries and later reads
//!   were rewritten to the wrong register;
//! - the read of the accumulator by `UMLAL/SMLAL` and `SMLALxy` was missed, so a write that set up
//!   the accumulator was deleted as dead;
//! - `JumpExchange { Reg(15) }` was treated as "reads PC" unconditionally, so `pop {pc}` — write
//!   PC, then jump, the standard shape of a Thumb function return — always fell back to
//!   single-stepping.
//!
//! Now there is exactly one copy: **adding an `IrOp` variant makes the compiler report a
//! non-exhaustive match here**, rather than letting some downstream consumer miss it.

use super::{Address, IrOp, Shifter, Value};

/// How many registers one operation can write: an `LDM` list (sixteen) plus a base writeback.
///
/// A buffer must be sized to this bound, because `writes_regs` **drops silently** when it runs out
/// of room, and a dropped write leaves a downstream consumer (the known-values table, liveness
/// analysis) holding a stale conclusion.
pub const MAX_WRITES: usize = 16;

/// Whether the operation reads `reg`.
///
/// Note that the list bits of `LDM`/`STM` mean opposite things: on a load a PC in the list is a
/// **write**, on a store it is a **read**.
pub fn reads_reg(op: &IrOp, reg: u8) -> bool {
    let value_reads = |value: &Value| match value {
        Value::Reg(r) => *r == reg,
        Value::Shifted { reg: r, by, .. } => {
            *r == reg || matches!(by, Shifter::Reg(s) if *s == reg)
        }
        Value::Imm(_) | Value::ImmC { .. } => false,
    };
    let address_reads = |address: &Address| match address {
        Address::Register { base, offset, .. } => *base == reg || value_reads(offset),
        Address::Absolute(_) => false,
    };
    match op {
        IrOp::Alu { a, b, .. } => value_reads(a) || b.is_some_and(|v| value_reads(&v)),
        // A long multiply or long halfword multiply with accumulation reads the **old** value of
        // dst(:dst+1): UMLAL/SMLAL add Rm×Rs into RdHi:RdLo, SMLALxy adds a halfword product.
        IrOp::Mul { a, b, .. } => *a == reg || *b == reg || op_addend_reads(op, reg),
        IrOp::MulLong {
            a,
            b,
            accumulate,
            lo,
            hi,
            ..
        } => *a == reg || *b == reg || (*accumulate && (*lo == reg || *hi == reg)),
        IrOp::MulHalf {
            dst,
            a,
            b,
            addend,
            long,
            ..
        } => {
            *a == reg
                || *b == reg
                || addend == &Some(reg)
                || (*long && (*dst == reg || dst.wrapping_add(1) == reg))
        }
        IrOp::MulWordHalf { a, b, addend, .. } => *a == reg || *b == reg || addend == &Some(reg),
        IrOp::Saturating { a, b, .. } => *a == reg || *b == reg,
        IrOp::Clz { src, .. } => *src == reg,
        IrOp::Swap { src, address, .. } => *src == reg || address_reads(address),
        IrOp::Load { address, .. } | IrOp::LoadDouble { address, .. } => address_reads(address),
        IrOp::Store { src, address, .. } => *src == reg || address_reads(address),
        IrOp::StoreDouble { rt, address, .. } => {
            *rt == reg || rt.wrapping_sub(1) == reg || address_reads(address)
        }
        // Load: a bit in the list is a **write** target; store: a bit in the list is a **read**,
        // since those registers' values have to be written out. Getting this asymmetry backwards
        // once meant that r0 in `stm {r0, r6, r8, r9}` did not count as a read, so the preceding
        // `mov r0, #1` was judged dead and deleted, and the function's return value vanished — on
        // a real host port, `AEEMod_Load` returned a garbage pointer.
        IrOp::LoadMultiple { base, .. } => *base == reg,
        IrOp::StoreMultiple { base, regs, .. } => *base == reg || regs & (1 << reg) != 0,
        IrOp::Jump { target } | IrOp::JumpExchange { target } => value_reads(target),
        IrOp::BranchConditional { target, .. } => value_reads(target),
        IrOp::WriteStatus { src, .. } => value_reads(src),
        IrOp::ReadStatus { .. } => false,
        IrOp::Nop | IrOp::Cond { .. } | IrOp::Breakpoint { .. } | IrOp::Swi { .. } => false,
    }
}

/// `Mul`'s accumulator register (`MLA`'s `Ra`). Split out only to keep the `match` above from
/// growing an overlarge pattern.
fn op_addend_reads(op: &IrOp, reg: u8) -> bool {
    matches!(op, IrOp::Mul { accumulate: Some(r), .. } if *r == reg)
}

/// Which registers the operation writes (returns the count and fills `out[..n]`).
pub fn writes_regs(op: &IrOp, out: &mut [u8]) -> usize {
    let mut n = 0;
    let push = |out: &mut [u8], n: &mut usize, r: u8| {
        if *n < out.len() {
            out[*n] = r;
            *n += 1;
        }
    };
    match op {
        IrOp::Alu { dst: Some(r), .. } => push(out, &mut n, *r),
        IrOp::ReadStatus { dst, .. } => push(out, &mut n, *dst),
        IrOp::Mul { dst, .. }
        | IrOp::MulWordHalf { dst, .. }
        | IrOp::Saturating { dst, .. }
        | IrOp::Clz { dst, .. } => push(out, &mut n, *dst),
        IrOp::MulLong { lo, hi, .. } => {
            push(out, &mut n, *lo);
            push(out, &mut n, *hi);
        }
        IrOp::MulHalf { dst, long, .. } => {
            push(out, &mut n, *dst);
            if *long {
                push(out, &mut n, dst.wrapping_add(1));
            }
        }
        IrOp::Swap { dst, .. } => push(out, &mut n, *dst),
        IrOp::Load {
            dst,
            writeback,
            address,
            ..
        } => {
            push(out, &mut n, *dst);
            if *writeback && let Address::Register { base, .. } = address {
                push(out, &mut n, *base);
            }
        }
        IrOp::LoadDouble {
            rt,
            writeback,
            address,
            ..
        } => {
            push(out, &mut n, *rt);
            push(out, &mut n, rt.wrapping_add(1));
            if *writeback && let Address::Register { base, .. } = address {
                push(out, &mut n, *base);
            }
        }
        IrOp::Store {
            writeback, address, ..
        }
        | IrOp::StoreDouble {
            writeback, address, ..
        } => {
            if *writeback && let Address::Register { base, .. } = address {
                push(out, &mut n, *base);
            }
        }
        IrOp::LoadMultiple {
            base,
            regs,
            writeback,
            ..
        } => {
            // A PC in the list counts as a write too: the `JumpExchange` that follows jumps there
            for r in 0..16u8 {
                if regs & (1 << r) != 0 {
                    push(out, &mut n, r);
                }
            }
            if *writeback || regs & (1 << base) != 0 {
                push(out, &mut n, *base);
            }
        }
        IrOp::StoreMultiple {
            base, writeback, ..
        } => {
            if *writeback {
                push(out, &mut n, *base);
            }
        }
        IrOp::Nop
        | IrOp::Cond { .. }
        | IrOp::Jump { .. }
        | IrOp::BranchConditional { .. }
        | IrOp::JumpExchange { .. }
        | IrOp::Breakpoint { .. }
        | IrOp::Swi { .. }
        | IrOp::WriteStatus { .. }
        | IrOp::Alu { dst: None, .. } => {}
    }
    n
}

/// Whether the operation writes `reg`.
pub fn writes_reg(op: &IrOp, reg: u8) -> bool {
    let mut out = [0u8; MAX_WRITES];
    let n = writes_regs(op, &mut out);
    out[..n].contains(&reg)
}

/// Whether the operation goes through a host callback, and so may report an access fault.
///
/// The JIT uses it to decide whether a block's prologue has to load the `memory` pointer; the
/// interpreter uses it to pick the set of operations that need a fault check afterwards.
pub fn may_fault(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::Load { .. }
            | IrOp::Store { .. }
            | IrOp::LoadDouble { .. }
            | IrOp::StoreDouble { .. }
            | IrOp::LoadMultiple { .. }
            | IrOp::StoreMultiple { .. }
            | IrOp::Swap { .. }
    )
}

/// Whether the instruction's operation sequence reads PC **before** it writes PC.
///
/// Inside a block r15 does not advance from instruction to instruction (it is written only at the
/// tail), so an instruction that reads PC part-way through would see a stale value and cannot go
/// into a block. But if the instruction writes PC itself first (`ldm {..,pc}` / `ldr pc, [..]` /
/// `mov pc, ..`), the later "read" reads what was just written — exactly the shape of a Thumb
/// function return (`pop {pc}`), which must be allowed to form a block, or every return point falls
/// back to single-stepping.
pub fn reads_pc_before_write(ops: &[IrOp]) -> bool {
    let mut pc_written = false;
    for op in ops {
        if reads_reg(op, 15) && !pc_written {
            return true;
        }
        if writes_reg(op, 15) {
            pc_written = true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AluOp, MemSize};

    fn alu(op: AluOp, dst: Option<u8>, a: Value, b: Option<Value>) -> IrOp {
        IrOp::Alu {
            op,
            dst,
            a,
            b,
            set_flags: false,
        }
    }

    fn regs(op: &IrOp) -> Vec<u8> {
        let mut out = [0u8; MAX_WRITES];
        let n = writes_regs(op, &mut out);
        out[..n].to_vec()
    }

    #[test]
    fn alu_reads_and_writes() {
        let op = alu(AluOp::Add, Some(0), Value::Reg(1), Some(Value::Imm(2)));
        assert!(reads_reg(&op, 1));
        assert!(!reads_reg(&op, 2));
        assert_eq!(regs(&op), vec![0]);
        // Test operations (CMP/TST) write no destination
        let op = alu(AluOp::Cmp, None, Value::Reg(1), Some(Value::Reg(2)));
        assert!(reads_reg(&op, 2));
        assert!(regs(&op).is_empty());
    }

    #[test]
    fn ldm_with_pc_writes_pc_but_does_not_read_it() {
        // `ldmia r0, {r1, pc}`: PC is a write target (the following JumpExchange is what reads it)
        let op = IrOp::LoadMultiple {
            base: 0,
            regs: (1 << 1) | (1 << 15),
            add: true,
            pre_index: false,
            writeback: false,
        };
        assert!(
            writes_reg(&op, 15),
            "a PC in the list has to count as a write"
        );
        assert!(!reads_reg(&op, 15));
        assert!(reads_reg(&op, 0), "the base register is a read");
        assert_eq!(regs(&op), vec![1, 15]);
    }

    #[test]
    fn stm_list_registers_are_reads() {
        // Every register in a store list has to be read out (this was once missed, and the return
        // value got deleted as a dead write)
        let op = IrOp::StoreMultiple {
            base: 4,
            regs: (1 << 0) | (1 << 6) | (1 << 8) | (1 << 9),
            add: true,
            pre_index: true,
            writeback: false,
        };
        for r in [0u8, 6, 8, 9] {
            assert!(
                reads_reg(&op, r),
                "r{r} is in the store list, so it must count as a read"
            );
        }
        assert!(!reads_reg(&op, 7));
        assert!(reads_reg(&op, 4), "the base is a read");
        assert!(
            writes_regs(&op, &mut [0u8; MAX_WRITES]) == 0,
            "with no writeback no register is written"
        );
    }

    #[test]
    fn stm_with_pc_reads_pc() {
        let op = IrOp::StoreMultiple {
            base: 0,
            regs: (1 << 1) | (1 << 15),
            add: true,
            pre_index: false,
            writeback: false,
        };
        assert!(reads_reg(&op, 15), "storing PC out is a read");
        assert!(!writes_reg(&op, 15));
    }

    #[test]
    fn ldm_can_write_all_sixteen_registers() {
        // A former bug: the buffer held four slots, so LDM's writes were silently truncated
        let op = IrOp::LoadMultiple {
            base: 3,
            regs: 0xFFFF,
            add: true,
            pre_index: false,
            writeback: true,
        };
        let w = regs(&op);
        assert_eq!(
            w.len(),
            16,
            "r0..r15 plus the base writeback, none may be lost"
        );
        for r in 0..=15u8 {
            assert!(w.contains(&r), "r{r} went missing");
        }
    }

    #[test]
    fn accumulate_reads_are_reported() {
        // UMLAL/SMLAL: the accumulator is read
        let op = IrOp::MulLong {
            lo: 2,
            hi: 3,
            a: 4,
            b: 5,
            signed: false,
            accumulate: true,
            set_flags: false,
        };
        assert!(reads_reg(&op, 2) && reads_reg(&op, 3));
        // Without accumulation they are write targets only
        let op = IrOp::MulLong {
            lo: 2,
            hi: 3,
            a: 4,
            b: 5,
            signed: false,
            accumulate: false,
            set_flags: false,
        };
        assert!(!reads_reg(&op, 2) && !reads_reg(&op, 3));
        // SMLALxy: dst:dst+1 is the accumulator
        let op = IrOp::MulHalf {
            dst: 6,
            addend: None,
            a: 0,
            b: 1,
            a_high: false,
            b_high: false,
            long: true,
            set_flags: false,
        };
        assert!(reads_reg(&op, 6) && reads_reg(&op, 7));
    }

    #[test]
    fn store_double_reads_both_halves() {
        let op = IrOp::StoreDouble {
            rt: 5,
            address: Address::Register {
                base: 0,
                offset: Value::Imm(8),
                add: true,
            },
            pre_index: true,
            writeback: false,
        };
        assert!(
            reads_reg(&op, 5) && reads_reg(&op, 4),
            "Rt and Rt-1 are both reads"
        );
    }

    #[test]
    fn pc_read_before_write_detects_the_common_forms() {
        // `ldr r0, [pc, #4]`: reads PC and never writes it, so it cannot go into a block
        let ldr_pc = IrOp::Load {
            dst: 0,
            address: Address::Register {
                base: 15,
                offset: Value::Imm(4),
                add: true,
            },
            size: MemSize::U32,
            signed: false,
            pre_index: true,
            writeback: false,
        };
        assert!(reads_pc_before_write(&[ldr_pc]));
        // `pop {pc}`: writes PC first, then jumps there, so it can go into a block
        let pop_pc = [
            IrOp::LoadMultiple {
                base: 13,
                regs: 1 << 15,
                add: true,
                pre_index: false,
                writeback: true,
            },
            IrOp::JumpExchange {
                target: Value::Reg(15),
            },
        ];
        assert!(!reads_pc_before_write(&pop_pc));
        // `bx pc`: read with no preceding write, so it cannot go into a block
        assert!(reads_pc_before_write(&[IrOp::JumpExchange {
            target: Value::Reg(15)
        }]));
    }

    #[test]
    fn may_fault_covers_exactly_the_memory_ops() {
        assert!(may_fault(&IrOp::Load {
            dst: 0,
            address: Address::Absolute(0x1000),
            size: MemSize::U32,
            signed: false,
            pre_index: true,
            writeback: false,
        }));
        assert!(may_fault(&IrOp::StoreMultiple {
            base: 0,
            regs: 1,
            add: true,
            pre_index: false,
            writeback: true,
        }));
        assert!(!may_fault(&alu(AluOp::Add, Some(0), Value::Imm(1), None)));
        assert!(!may_fault(&IrOp::Swi { imm: 0 }));
    }
}
