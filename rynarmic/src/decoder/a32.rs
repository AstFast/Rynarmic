//! The A32 decoder: ARM state, 32-bit fixed-length encodings.
//!
//! Target ISA: **ARMv5TEJ**, the ARM926EJ-S level.
//! Covered: the integer core; v5's CLZ, BKPT and BLX; v5TE's halfword and signed accesses, 64-bit multiplies,
//! halfword multiplies, saturating arithmetic, and LDRD/STRD.
//! Not covered: VFP/NEON, Jazelle bytecode execution (`BXJ` is treated as `BX`, matching the hardware's fallback behaviour),
//! and the v6 and later media and exclusive-access instructions.

use crate::common::bits as bitops;
use crate::common::bits::{ShiftType, shift_c, sign_extend};
use crate::decoder::DecodeError;
use crate::ir::{Address, AluOp, IrOp, MemSize, SatOp, Shifter, Value};

/// Decodes one A32 instruction.
///
/// - `pc`: this instruction's own address, used for PC-relative arithmetic (the ARM pipeline offset is +8)
/// - `cond_passed`: whether the condition code holds; when it does not, an empty operation list comes back and the instruction is skipped
pub fn decode(inst: u32, pc: u32, cond_passed: bool) -> Result<Vec<IrOp>, DecodeError> {
    let mut ops = Vec::with_capacity(2);
    decode_into(inst, pc, cond_passed, &mut ops)?;
    Ok(ops)
}

/// As [`decode`], but appending into a caller-supplied buffer so that no per-instruction allocation happens.
pub fn decode_into(
    inst: u32,
    pc: u32,
    cond_passed: bool,
    ops: &mut Vec<IrOp>,
) -> Result<(), DecodeError> {
    if !cond_passed {
        return Ok(());
    }

    let cond = inst >> 28;
    if cond == 0xF {
        return unconditional(inst, pc, ops);
    }

    match bitops::bits(inst, 27, 25) {
        0b000 | 0b001 => misc_and_data_processing(inst, pc, ops)?,
        0b010 => load_store(inst, pc, ops, false)?,
        0b011 => {
            if bitops::bit(inst, 4) == 1 {
                return Err(DecodeError::Unsupported {
                    encoding: inst,
                    reason: "media instructions (ARMv6)",
                });
            }
            load_store(inst, pc, ops, true)?;
        }
        0b100 => block_transfer(inst, ops)?,
        0b101 => branch(inst, pc, ops),
        // 0b110 and 0b111 are the coprocessor space, of which 1111 is SWI
        0b111 if bitops::bits(inst, 27, 24) == 0b1111 => {
            ops.push(IrOp::Swi {
                imm: inst & 0x00FF_FFFF,
            });
        }
        _ => {
            return Err(DecodeError::Unsupported {
                encoding: inst,
                reason: "coprocessor (VFP/NEON) instruction",
            });
        }
    }

    Ok(())
}

/// Reading PC in A32 state gives the instruction address plus 8, the pipeline offset.
#[inline]
fn pc_value(pc: u32) -> u32 {
    pc.wrapping_add(8)
}

/// Folds `[pc, +/- delta]` into an absolute address. Folding at translation time keeps a cached block correct.
#[inline]
fn pc_relative(pc: u32, delta: u32, add: bool) -> u32 {
    let base = pc_value(pc);
    if add {
        base.wrapping_add(delta)
    } else {
        base.wrapping_sub(delta)
    }
}

/// The shift type in bits 6-5.
#[inline]
fn shift_type(inst: u32) -> ShiftType {
    match bitops::bits(inst, 6, 5) {
        0b00 => ShiftType::Lsl,
        0b01 => ShiftType::Lsr,
        0b10 => ShiftType::Asr,
        _ => ShiftType::Ror,
    }
}

/// Decodes the immediate shift amount for bit4 == 0.
///
/// An immediate of 0 means different things per type: only LSL means no shift; LSR and ASR mean 32, ROR means RRX.
#[inline]
fn immediate_shift(inst: u32) -> (ShiftType, u8) {
    let mut shift = shift_type(inst);
    let mut amount = bitops::bits(inst, 11, 7) as u8;
    if amount == 0 {
        match shift {
            ShiftType::Lsr | ShiftType::Asr => amount = 32,
            ShiftType::Ror => shift = ShiftType::Rrx,
            ShiftType::Lsl | ShiftType::Rrx => {}
        }
    }
    (shift, amount)
}

/// The unconditional space, cond == 0b1111: in ARMv5 it holds only BLX(imm) and PLD.
fn unconditional(inst: u32, pc: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    // BLX (imm): 1111 101H imm24, jumping and switching to Thumb state
    if bitops::bits(inst, 27, 25) == 0b101 {
        let offset = (sign_extend(inst & 0x00FF_FFFF, 24) << 2) | (bitops::bit(inst, 24) << 1);
        let target = pc.wrapping_add(8).wrapping_add(offset);
        ops.push(IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(14),
            a: Value::Imm(pc.wrapping_add(4)),
            b: None,
            set_flags: false,
        });
        // A low bit of 1 in the target means entering Thumb
        ops.push(IrOp::JumpExchange {
            target: Value::Imm(target | 1),
        });
        return Ok(());
    }

    // PLD (imm/reg): a pure hint, equivalent to Nop in execution
    if bitops::bits(inst, 27, 26) == 0b01 && bitops::bits(inst, 15, 12) == 0xF {
        ops.push(IrOp::Nop);
        return Ok(());
    }

    Err(DecodeError::Unsupported {
        encoding: inst,
        reason: "unconditional instruction (ARMv6+/v7)",
    })
}

/// The op1 == 0b000/0b001 space, which mixes branch-and-exchange, multiplies, access extensions, PSR and data processing.
fn misc_and_data_processing(inst: u32, pc: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    // BX / BLX(reg) / BXJ: 0001 0010 1111 1111 1111 0001|0011|0010 mmmm
    if bitops::bits(inst, 27, 20) == 0b0001_0010 && bitops::bits(inst, 19, 8) == 0xFFF {
        let rm = bitops::bits(inst, 3, 0) as u8;
        match bitops::bits(inst, 7, 4) {
            0b0001 => ops.push(IrOp::JumpExchange {
                target: Value::Reg(rm),
            }),
            0b0011 => {
                ops.push(IrOp::Alu {
                    op: AluOp::Mov,
                    dst: Some(14),
                    a: Value::Imm(pc.wrapping_add(4)),
                    b: None,
                    set_flags: false,
                });
                ops.push(IrOp::JumpExchange {
                    target: Value::Reg(rm),
                });
            }
            // v5TEJ's BXJ: Jazelle is not implemented, so it falls back to ARM state with BX's semantics
            0b0010 => ops.push(IrOp::JumpExchange {
                target: Value::Reg(rm),
            }),
            _ => return Err(DecodeError::Undefined(inst)),
        }
        return Ok(());
    }

    let field20 = bitops::bits(inst, 27, 20);
    let low = bitops::bits(inst, 11, 4);

    // Multiply and long multiply: both end in 1001
    if bitops::bits(inst, 27, 22) == 0 && bitops::bits(inst, 7, 4) == 0b1001 {
        return multiply(inst, ops);
    }
    if bitops::bits(inst, 27, 23) == 0b0_0001 && bitops::bits(inst, 7, 4) == 0b1001 {
        return multiply_long(inst, ops);
    }

    // BKPT: 0001 0010 imm12 0111 imm4
    if field20 == 0x12 && bitops::bits(inst, 7, 4) == 0b0111 {
        let imm = (bitops::bits(inst, 19, 8) << 4) | bitops::bits(inst, 3, 0);
        ops.push(IrOp::Breakpoint { imm });
        return Ok(());
    }

    // SWP / SWPB: 0001 0000|0100 Rn Rd 0000 1001 Rm
    if bitops::bits(inst, 27, 23) == 0b0_0010 && low == 0b0000_1001 {
        let byte = bitops::bit(inst, 22) == 1;
        ops.push(IrOp::Swap {
            dst: bitops::bits(inst, 15, 12) as u8,
            src: bitops::bits(inst, 3, 0) as u8,
            address: Address::Register {
                base: bitops::bits(inst, 19, 16) as u8,
                offset: Value::Imm(0),
                add: true,
            },
            byte,
        });
        return Ok(());
    }

    // QADD/QSUB/QDADD/QDSUB: 0001 0000|0010|0100|0110 Rn Rd 0000 0101 Rm
    if low == 0b0000_0101 {
        let sat = match field20 {
            0x10 => Some(SatOp::Add),
            0x12 => Some(SatOp::Sub),
            0x14 => Some(SatOp::DoubleAdd),
            0x16 => Some(SatOp::DoubleSub),
            _ => None,
        };
        if let Some(op) = sat {
            ops.push(IrOp::Saturating {
                op,
                dst: bitops::bits(inst, 15, 12) as u8,
                a: bitops::bits(inst, 19, 16) as u8,
                b: bitops::bits(inst, 3, 0) as u8,
            });
            return Ok(());
        }
    }

    // CLZ: 0001 0110 1111 Rd 1111 0001 Rm
    if field20 == 0x16 && bitops::bits(inst, 19, 16) == 0xF && low == 0b1111_0001 {
        ops.push(IrOp::Clz {
            dst: bitops::bits(inst, 15, 12) as u8,
            src: bitops::bits(inst, 3, 0) as u8,
        });
        return Ok(());
    }

    // ARMv5TE halfword multiplies: SMLAxy, SMULxy, SMLALxy, SMLAWy and SMULWy
    if bitops::bit(inst, 4) == 0
        && bitops::bit(inst, 7) == 1
        && let Some(op) = halfword_multiply(inst)?
    {
        ops.push(op);
        return Ok(());
    }

    // MRS: 0001 0000 1111 Rd 0000 0000 0000
    if field20 == 0x10 && bitops::bits(inst, 19, 16) == 0xF && bitops::bits(inst, 11, 0) == 0 {
        ops.push(IrOp::ReadStatus {
            dst: bitops::bits(inst, 15, 12) as u8,
            spsr: bitops::bit(inst, 22) == 1,
        });
        return Ok(());
    }

    // MSR (reg): 0001 0010 mask 1111 0000 0000 Rm
    if field20 == 0x12 && bitops::bits(inst, 15, 12) == 0xF && bitops::bits(inst, 11, 4) == 0 {
        ops.push(IrOp::WriteStatus {
            src: Value::Reg(bitops::bits(inst, 3, 0) as u8),
            mask: psr_mask(inst),
            spsr: bitops::bit(inst, 22) == 1,
        });
        return Ok(());
    }

    // MSR (imm): 0011 0010 mask 1111 rotate imm8
    if field20 == 0x32 && bitops::bits(inst, 15, 12) == 0xF {
        let (value, _) = bitops::expand_imm_c(inst & 0xFFF, false);
        ops.push(IrOp::WriteStatus {
            src: Value::Imm(value),
            mask: psr_mask(inst),
            spsr: bitops::bit(inst, 22) == 1,
        });
        return Ok(());
    }

    // Halfword, signed and doubleword accesses.
    // The discriminator for the extra access forms is bit7 (always 0 in a register-shifted data-processing encoding),
    // together with bit4 == 1; without it a register-shifted `mov r0, r1, lsr r2` would be taken for a halfword access.
    if bitops::bit(inst, 25) == 0
        && bitops::bit(inst, 7) == 1
        && bitops::bit(inst, 4) == 1
        && bitops::bits(inst, 6, 5) != 0
    {
        return extra_load_store(inst, pc, ops);
    }

    data_processing(inst, pc, ops)
}

fn psr_mask(inst: u32) -> u32 {
    // The four mask bits are f/s/x/c and map onto CPSR's bit fields
    let mask = bitops::bits(inst, 19, 16);
    let mut result = 0;
    if mask & 0b1000 != 0 {
        result |= 0xFF00_0000; // the condition flags
    }
    if mask & 0b0100 != 0 {
        result |= 0x00FF_0000; // the status bits (T, J, IT and so on)
    }
    if mask & 0b0010 != 0 {
        result |= 0x0000_FF00; // the extension bits
    }
    if mask & 0b0001 != 0 {
        result |= 0x0000_00FF; // the control bits (mode and interrupt masks)
    }
    result
}

fn multiply(inst: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let accumulate = bitops::bit(inst, 21) == 1;
    let set_flags = bitops::bit(inst, 20) == 1;
    let rd = bitops::bits(inst, 19, 16) as u8;
    let ra = bitops::bits(inst, 15, 12) as u8;
    let rm = bitops::bits(inst, 11, 8) as u8;
    let rs = bitops::bits(inst, 3, 0) as u8;

    ops.push(IrOp::Mul {
        dst: rd,
        a: rm,
        b: rs,
        accumulate: accumulate.then_some(ra),
        set_flags,
    });
    Ok(())
}

fn multiply_long(inst: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let signed = bitops::bit(inst, 22) == 1;
    let accumulate = bitops::bit(inst, 21) == 1;
    let set_flags = bitops::bit(inst, 20) == 1;
    let hi = bitops::bits(inst, 19, 16) as u8;
    let lo = bitops::bits(inst, 15, 12) as u8;
    let rm = bitops::bits(inst, 11, 8) as u8;
    let rs = bitops::bits(inst, 3, 0) as u8;

    if hi == 15 || lo == 15 {
        return Err(DecodeError::Unsupported {
            encoding: inst,
            reason: "long multiply writing PC",
        });
    }
    ops.push(IrOp::MulLong {
        lo,
        hi,
        a: rm,
        b: rs,
        signed,
        accumulate,
        set_flags,
    });
    Ok(())
}

fn halfword_multiply(inst: u32) -> Result<Option<IrOp>, DecodeError> {
    let field20 = bitops::bits(inst, 27, 20);
    let rd = bitops::bits(inst, 19, 16) as u8;
    let ra = bitops::bits(inst, 15, 12) as u8;
    let rm = bitops::bits(inst, 11, 8) as u8;
    let rs = bitops::bits(inst, 3, 0) as u8;
    // The halfword select bits, established by probing a reference implementation:
    // x is bit5 and always selects the halfword of bits(3,0), i.e. Rn, the second syntactic operand;
    // y is bit6 and always selects the halfword of bits(11,8), i.e. Rm, the third syntactic operand.
    // In the IR, MulHalf has a = rm and b = rs, so a_high takes y and b_high takes x.
    let x_half = bitops::bit(inst, 5) == 1;
    let y_half = bitops::bit(inst, 6) == 1;

    match field20 {
        // SMLAxy: Rd = Ra + Rn.x * Rm.y (NZCV untouched; an accumulation overflow sets Q, which the executor handles)
        0x10 => Ok(Some(IrOp::MulHalf {
            dst: rd,
            addend: Some(ra),
            a: rm,
            b: rs,
            a_high: y_half,
            b_high: x_half,
            long: false,
            set_flags: false,
        })),
        // SMULxy: Rd = Rn.x * Rm.y (Ra is fixed at 0 and no flags are set)
        0x16 if ra == 0 => Ok(Some(IrOp::MulHalf {
            dst: rd,
            addend: None,
            a: rm,
            b: rs,
            a_high: y_half,
            b_high: x_half,
            long: false,
            set_flags: false,
        })),
        // SMLALxy: RdHi:RdLo += Rn.x * Rm.y (RdLo must be even and RdLo+1 <= 14)
        0x14 if ra.is_multiple_of(2) && ra <= 12 => Ok(Some(IrOp::MulHalf {
            dst: ra,
            addend: None,
            a: rm,
            b: rs,
            a_high: y_half,
            b_high: x_half,
            long: true,
            set_flags: false,
        })),
        // SMLALxy with RdLo out of range: an undefined encoding
        0x14 => Err(DecodeError::Undefined(inst)),
        // SMLAWy: Rd = Ra + ((Rn * Rm.y) >> 16), where the word comes from bits(3,0) and the y halfword from bits(11,8)
        0x12 if bitops::bits(inst, 5, 4) == 0b00 => Ok(Some(IrOp::MulWordHalf {
            dst: rd,
            addend: Some(ra),
            a: rs,
            b: rm,
            b_high: y_half,
            set_flags: false,
        })),
        // SMULWy: Rd = (Rn * Rm.y) >> 16 (ra fixed at 0)
        0x12 if bitops::bits(inst, 5, 4) == 0b10 && ra == 0 => Ok(Some(IrOp::MulWordHalf {
            dst: rd,
            addend: None,
            a: rs,
            b: rm,
            b_high: y_half,
            set_flags: false,
        })),
        _ => Ok(None),
    }
}

fn extra_load_store(inst: u32, pc: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let pre_index = bitops::bit(inst, 24) == 1;
    let add = bitops::bit(inst, 23) == 1;
    let immediate = bitops::bit(inst, 22) == 1;
    let writeback_bit = bitops::bit(inst, 21) == 1;
    let load = bitops::bit(inst, 20) == 1;
    let rn = bitops::bits(inst, 19, 16) as u8;
    let rt = bitops::bits(inst, 15, 12) as u8;
    let op4 = bitops::bits(inst, 7, 4);

    let offset = if immediate {
        Value::Imm((bitops::bits(inst, 11, 8) << 4) | bitops::bits(inst, 3, 0))
    } else {
        if bitops::bits(inst, 11, 8) != 0 {
            return Err(DecodeError::Undefined(inst));
        }
        Value::Reg(bitops::bits(inst, 3, 0) as u8)
    };

    // PC-relative addressing is folded into an absolute address at translation time, keeping a cached block correct
    let address = match (rn, offset) {
        (15, Value::Imm(delta)) => Address::Absolute(pc_relative(pc, delta, add)),
        _ => Address::Register {
            base: rn,
            offset,
            add,
        },
    };
    let writeback = if pre_index { writeback_bit } else { true };

    // A doubleword access requires an even Rt, and Rt+1 must not run past r14
    let double_ok = rt.is_multiple_of(2) && rt <= 13;

    match (op4, load) {
        // STRH / LDRH
        (0b1011, false) => ops.push(IrOp::Store {
            src: rt,
            address,
            size: MemSize::U16,
            pre_index,
            writeback,
        }),
        (0b1011, true) => ops.push(IrOp::Load {
            dst: rt,
            address,
            size: MemSize::U16,
            signed: false,
            pre_index,
            writeback,
        }),
        // LDRD (v5TE): two consecutive words
        (0b1101, false) => {
            if !double_ok {
                return Err(DecodeError::Undefined(inst));
            }
            ops.push(IrOp::LoadDouble {
                rt,
                address,
                pre_index,
                writeback,
            })
        }
        // STRD (v5TE)
        (0b1111, false) => {
            if !double_ok {
                return Err(DecodeError::Undefined(inst));
            }
            ops.push(IrOp::StoreDouble {
                rt,
                address,
                pre_index,
                writeback,
            })
        }
        // LDRSB / LDRSH
        (0b1101, true) => ops.push(IrOp::Load {
            dst: rt,
            address,
            size: MemSize::U8,
            signed: true,
            pre_index,
            writeback,
        }),
        (0b1111, true) => ops.push(IrOp::Load {
            dst: rt,
            address,
            size: MemSize::U16,
            signed: true,
            pre_index,
            writeback,
        }),
        _ => return Err(DecodeError::Undefined(inst)),
    }

    if load && rt == 15 {
        ops.push(IrOp::JumpExchange {
            target: Value::Reg(15),
        });
    }
    Ok(())
}

fn data_processing(inst: u32, pc: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let immediate = bitops::bit(inst, 25) == 1;
    let opcode = bitops::bits(inst, 24, 21);
    let set_flags = bitops::bit(inst, 20) == 1;
    let rn = bitops::bits(inst, 19, 16) as u8;
    let rd = bitops::bits(inst, 15, 12) as u8;

    let op = AluOp::from_opcode(opcode).ok_or(DecodeError::Undefined(inst))?;

    let operand2 = if immediate {
        let (value, carry) = bitops::expand_imm_c(inst & 0xFFF, false);
        if bitops::bits(inst, 11, 8) == 0 {
            Value::Imm(value)
        } else {
            Value::ImmC { value, carry }
        }
    } else {
        let rm = bitops::bits(inst, 3, 0) as u8;
        if bitops::bit(inst, 4) == 1 {
            // A register-specified shift amount: bits6-5 still select the shift type, and all four are legal
            let rs = bitops::bits(inst, 11, 8) as u8;
            Value::Shifted {
                reg: rm,
                shift: shift_type(inst),
                by: Shifter::Reg(rs),
            }
        } else {
            let (shift, amount) = immediate_shift(inst);
            // A PC shifted source is folded at translation time too, carry included
            if rm == 15 {
                let source = pc_value(pc);
                if shift == ShiftType::Rrx {
                    // RRX depends on the run-time carry and cannot be folded; the PC guard sends it back to single-stepping
                    Value::Shifted {
                        reg: 15,
                        shift,
                        by: Shifter::Imm(0),
                    }
                } else if amount == 0 {
                    Value::Imm(source)
                } else {
                    let (value, carry) = shift_c(source, shift, amount as u32, false);
                    Value::ImmC { value, carry }
                }
            } else {
                Value::Shifted {
                    reg: rm,
                    shift,
                    by: Shifter::Imm(amount),
                }
            }
        }
    };

    // A PC first operand is folded into an immediate at translation time, since r15 does not advance inside a block
    let (a, b) = if op.is_move() {
        (operand2, None)
    } else if rn == 15 {
        (Value::Imm(pc_value(pc)), Some(operand2))
    } else {
        (Value::Reg(rn), Some(operand2))
    };

    let dst = if op.is_test() { None } else { Some(rd) };
    ops.push(IrOp::Alu {
        op,
        dst,
        a,
        b,
        set_flags: set_flags || op.is_test(),
    });

    if dst == Some(15) {
        if set_flags {
            return Err(DecodeError::Unsupported {
                encoding: inst,
                reason: "exception return (data-processing write to PC with S=1)",
            });
        }
        ops.push(IrOp::JumpExchange {
            target: Value::Reg(15),
        });
    }

    Ok(())
}

fn load_store(
    inst: u32,
    pc: u32,
    ops: &mut Vec<IrOp>,
    register_form: bool,
) -> Result<(), DecodeError> {
    let pre_index = bitops::bit(inst, 24) == 1;
    let add = bitops::bit(inst, 23) == 1;
    let byte = bitops::bit(inst, 22) == 1;
    let writeback_bit = bitops::bit(inst, 21) == 1;
    let load = bitops::bit(inst, 20) == 1;
    let rn = bitops::bits(inst, 19, 16) as u8;
    let rd = bitops::bits(inst, 15, 12) as u8;

    let offset = if register_form {
        let rm = bitops::bits(inst, 3, 0) as u8;
        let (shift, amount) = immediate_shift(inst);
        Value::Shifted {
            reg: rm,
            shift,
            by: Shifter::Imm(amount),
        }
    } else {
        Value::Imm(inst & 0xFFF)
    };

    let address = match (rn, offset) {
        (15, Value::Imm(delta)) => Address::Absolute(pc_relative(pc, delta, add)),
        _ => Address::Register {
            base: rn,
            offset,
            add,
        },
    };
    let size = if byte { MemSize::U8 } else { MemSize::U32 };
    let writeback = if pre_index { writeback_bit } else { true };

    if load {
        ops.push(IrOp::Load {
            dst: rd,
            address,
            size,
            signed: false,
            pre_index,
            writeback,
        });
        if rd == 15 {
            ops.push(IrOp::JumpExchange {
                target: Value::Reg(15),
            });
        }
    } else {
        ops.push(IrOp::Store {
            src: rd,
            address,
            size,
            pre_index,
            writeback,
        });
    }

    Ok(())
}

fn block_transfer(inst: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let pre_index = bitops::bit(inst, 24) == 1;
    let add = bitops::bit(inst, 23) == 1;
    let user_bank = bitops::bit(inst, 22) == 1;
    let writeback = bitops::bit(inst, 21) == 1;
    let load = bitops::bit(inst, 20) == 1;
    let rn = bitops::bits(inst, 19, 16) as u8;
    let regs = bitops::bits(inst, 15, 0) as u16;

    if user_bank {
        return Err(DecodeError::Unsupported {
            encoding: inst,
            reason: "LDM/STM with user-mode bank (S=1)",
        });
    }
    if regs == 0 {
        return Err(DecodeError::Undefined(inst));
    }

    if load {
        ops.push(IrOp::LoadMultiple {
            base: rn,
            regs,
            add,
            pre_index,
            writeback,
        });
        if regs & (1 << 15) != 0 {
            ops.push(IrOp::JumpExchange {
                target: Value::Reg(15),
            });
        }
    } else {
        ops.push(IrOp::StoreMultiple {
            base: rn,
            regs,
            add,
            pre_index,
            writeback,
        });
    }

    Ok(())
}

fn branch(inst: u32, pc: u32, ops: &mut Vec<IrOp>) {
    let link = bitops::bit(inst, 24) == 1;
    let offset = sign_extend(inst & 0x00FF_FFFF, 24) << 2;
    let target = pc.wrapping_add(8).wrapping_add(offset);

    if link {
        ops.push(IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(14),
            a: Value::Imm(pc.wrapping_add(4)),
            b: None,
            set_flags: false,
        });
    }
    ops.push(IrOp::Jump {
        target: Value::Imm(target),
    });
}

/// Whether the encoding is B or BL; a conditional branch can safely go into a cached block.
pub fn is_branch(inst: u32) -> bool {
    bitops::bits(inst, 27, 25) == 0b101
}

/// Decodes a conditional branch. The condition is evaluated at **run time**, which is what lets the block be cached and re-run.
pub fn decode_branch_conditional(inst: u32, pc: u32, cond: u32, ops: &mut Vec<IrOp>) {
    let link = bitops::bit(inst, 24) == 1;
    let offset = sign_extend(inst & 0x00FF_FFFF, 24) << 2;
    let target = pc.wrapping_add(8).wrapping_add(offset);

    if link {
        // A failed `BL<cond>` **may not write LR**: the LR write has to share the branch's condition.
        ops.push(IrOp::Cond { cond, skip: 1 });
        ops.push(IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(14),
            a: Value::Imm(pc.wrapping_add(4)),
            b: None,
            set_flags: false,
        });
    }
    ops.push(IrOp::BranchConditional {
        cond,
        target: Value::Imm(target),
    });
}
