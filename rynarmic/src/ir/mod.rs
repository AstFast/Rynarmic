//! The intermediate representation.
//!
//! As in dynarmic, the decoder does not execute anything directly: it first translates one guest
//! instruction into a handful of IR operations, and the interpreter or the JIT backend consumes
//! those. The benefit is that a backend can reuse the IR-level optimizations (constant folding,
//! dead-code elimination, block linking) and that further backends have somewhere to plug in.

pub use crate::common::bits::ShiftType;

pub mod facts;
mod optimize;
pub use optimize::{optimize, totals as optimize_totals};

/// Where a shift amount comes from: an immediate, or a register (`LSL r0, r1, r2` and friends).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shifter {
    Imm(u8),
    Reg(u8),
}

impl Shifter {
    pub const NONE: Shifter = Shifter::Imm(0);
}

/// An IR operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Reg(u8),
    Imm(u32),
    /// An immediate whose carry is already known (A32's rotated immediate, with rotate != 0).
    ///
    /// Logical operations use it to update C; arithmetic operations still go by their result.
    ImmC {
        value: u32,
        carry: bool,
    },
    Shifted {
        reg: u8,
        shift: ShiftType,
        by: Shifter,
    },
}

impl Value {
    pub const fn reg(index: u8) -> Value {
        Value::Reg(index)
    }

    pub const fn imm(value: u32) -> Value {
        Value::Imm(value)
    }
}

/// Data-processing operations (the sixteen A32 opcodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AluOp {
    And,
    Eor,
    Sub,
    Rsb,
    Add,
    Adc,
    Sbc,
    Rsc,
    Tst,
    Teq,
    Cmp,
    Cmn,
    Orr,
    Mov,
    Bic,
    Mvn,
}

impl AluOp {
    /// Whether this opcode only updates the flags and writes no result back.
    pub const fn is_test(self) -> bool {
        matches!(self, AluOp::Tst | AluOp::Teq | AluOp::Cmp | AluOp::Cmn)
    }

    /// `MOV`/`MVN` use the first operand only.
    pub const fn is_move(self) -> bool {
        matches!(self, AluOp::Mov | AluOp::Mvn)
    }

    /// Logical operations: they update N/Z/C and leave V alone.
    pub const fn is_logical(self) -> bool {
        matches!(
            self,
            AluOp::And
                | AluOp::Eor
                | AluOp::Orr
                | AluOp::Bic
                | AluOp::Mov
                | AluOp::Mvn
                | AluOp::Tst
                | AluOp::Teq
        )
    }

    pub fn from_opcode(opcode: u32) -> Option<AluOp> {
        Some(match opcode {
            0b0000 => AluOp::And,
            0b0001 => AluOp::Eor,
            0b0010 => AluOp::Sub,
            0b0011 => AluOp::Rsb,
            0b0100 => AluOp::Add,
            0b0101 => AluOp::Adc,
            0b0110 => AluOp::Sbc,
            0b0111 => AluOp::Rsc,
            0b1000 => AluOp::Tst,
            0b1001 => AluOp::Teq,
            0b1010 => AluOp::Cmp,
            0b1011 => AluOp::Cmn,
            0b1100 => AluOp::Orr,
            0b1101 => AluOp::Mov,
            0b1110 => AluOp::Bic,
            0b1111 => AluOp::Mvn,
            _ => return None,
        })
    }
}

/// Access width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemSize {
    U8,
    U16,
    U32,
}

impl MemSize {
    pub const fn bytes(self) -> u32 {
        match self {
            MemSize::U8 => 1,
            MemSize::U16 => 2,
            MemSize::U32 => 4,
        }
    }
}

/// An access address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    /// `[base, +/- offset]`, with the base in a register.
    Register { base: u8, offset: Value, add: bool },
    /// An absolute address.
    ///
    /// PC-relative addressing (`ldr r0, [pc, #8]`, Thumb's literal loads) is folded into an
    /// absolute address at **translation time**, which is what keeps a cached block correct: r15
    /// does not advance from instruction to instruction inside a block.
    Absolute(u32),
}

/// Saturating arithmetic (ARMv5TE's QADD/QSUB/QDADD/QDSUB).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SatOp {
    Add,
    Sub,
    DoubleAdd,
    DoubleSub,
}

/// One IR operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrOp {
    /// A no-op (a PLD prefetch hint, for instance).
    Nop,
    /// Data processing. `Mov`/`Mvn` look at `a` only; test operations ignore the write-back.
    Alu {
        op: AluOp,
        dst: Option<u8>,
        a: Value,
        b: Option<Value>,
        set_flags: bool,
    },
    /// 32-bit multiply (MUL/MLA). `accumulate` is MLA's accumulator register.
    Mul {
        dst: u8,
        a: u8,
        b: u8,
        accumulate: Option<u8>,
        set_flags: bool,
    },
    /// A 64-bit multiply result (UMULL/UMLAL/SMULL/SMLAL), written to RdLo/RdHi. `accumulate`
    /// requests the 64-bit accumulation.
    MulLong {
        lo: u8,
        hi: u8,
        a: u8,
        b: u8,
        signed: bool,
        accumulate: bool,
        set_flags: bool,
    },
    /// Halfword multiply (SMULxy/SMLAxy/SMLALxy). `a_high`/`b_high` select the top half of each
    /// operand; with `long`, the result accumulates into the 64-bit value in `dst`:`dst+1`
    /// (SMLALxy).
    MulHalf {
        dst: u8,
        addend: Option<u8>,
        a: u8,
        b: u8,
        a_high: bool,
        b_high: bool,
        long: bool,
        set_flags: bool,
    },
    /// Word-by-halfword multiply (SMULWy/SMLAWy).
    MulWordHalf {
        dst: u8,
        addend: Option<u8>,
        a: u8,
        b: u8,
        b_high: bool,
        set_flags: bool,
    },
    /// Saturating arithmetic; on overflow it sets CPSR.Q.
    Saturating { op: SatOp, dst: u8, a: u8, b: u8 },
    /// Count leading zeros (CLZ).
    Clz { dst: u8, src: u8 },
    /// Atomic swap (SWP/SWPB).
    Swap {
        dst: u8,
        src: u8,
        address: Address,
        byte: bool,
    },
    /// Breakpoint (BKPT).
    Breakpoint { imm: u32 },
    /// Load. `signed` covers LDRSB/LDRSH.
    Load {
        dst: u8,
        address: Address,
        size: MemSize,
        signed: bool,
        pre_index: bool,
        writeback: bool,
    },
    /// Store.
    Store {
        src: u8,
        address: Address,
        size: MemSize,
        pre_index: bool,
        writeback: bool,
    },
    /// LDRD (v5TE): load two consecutive words into `rt` and `rt + 1`.
    LoadDouble {
        rt: u8,
        address: Address,
        pre_index: bool,
        writeback: bool,
    },
    /// STRD (v5TE): store `rt` and `rt + 1` consecutively.
    StoreDouble {
        rt: u8,
        address: Address,
        pre_index: bool,
        writeback: bool,
    },
    /// LDM.
    LoadMultiple {
        base: u8,
        regs: u16,
        add: bool,
        pre_index: bool,
        writeback: bool,
    },
    /// STM.
    StoreMultiple {
        base: u8,
        regs: u16,
        add: bool,
        pre_index: bool,
        writeback: bool,
    },
    /// A condition guard: when the condition does not hold, skip the following `skip` operations.
    ///
    /// Every A32 instruction carries a 4-bit condition code, so a conditionally executed
    /// **non-branch** instruction (`MOVEQ`, `LDREQ`, …) expands into a `Cond` plus some operations.
    /// The condition is then evaluated at run time and the block stays cacheable; baking the
    /// condition into the block would take the wrong path on the second run, once the flags have
    /// changed.
    ///
    /// Note that `skip` is computed at translation time, so **a `Cond` and the operations it
    /// guards must be kept together**: nothing that deletes operations inside a block may split
    /// them apart.
    Cond { cond: u32, skip: u8 },
    /// A direct jump (B/BL have already written the return address, as has `MOV pc, ...`).
    Jump { target: Value },
    /// A jump whose condition is evaluated at run time: once the block is cached, it still decides
    /// afresh each time, from the flags as they stand.
    BranchConditional { cond: u32, target: Value },
    /// Branch and exchange instruction set (BX/BLX/BXJ); the target's low bit selects Thumb.
    JumpExchange { target: Value },
    /// MRS: read a PSR into a register.
    ReadStatus { dst: u8, spsr: bool },
    /// MSR: write a PSR. `mask` selects the writable fields bit by bit.
    WriteStatus { src: Value, mask: u32, spsr: bool },
    /// A software interrupt.
    Swi { imm: u32 },
}

impl IrOp {
    /// Whether this operation changes control flow (a block terminator).
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            IrOp::Jump { .. }
                | IrOp::BranchConditional { .. }
                | IrOp::JumpExchange { .. }
                | IrOp::Swi { .. }
                | IrOp::Breakpoint { .. }
        )
    }
}
