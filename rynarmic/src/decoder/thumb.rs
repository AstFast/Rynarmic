//! The Thumb decoder: ARMv5T Thumb-1.
//!
//! Covered: shifts; MOV/CMP/ADD/SUB immediates; the ALU group; high-register operations and
//! BX/BLX; LDR/STR (immediate, register, SP-relative, PC literal), LDRH/STRH, PUSH/POP, LDM/STM,
//! the two-halfword `BL`/`BLX` sequences, BKPT, conditional and unconditional branches, and SWI.
//!
//! Not covered: byte-reverse (ARMv6), CBZ and the like; hitting one returns `Unsupported`.

use crate::common::bits as bitops;
use crate::common::bits::{ShiftType, sign_extend};
use crate::decoder::DecodeError;
use crate::ir::{Address, AluOp, IrOp, MemSize, Shifter, Value};

pub fn decode(inst: u16, pc: u32, cond_passed: bool) -> Result<Vec<IrOp>, DecodeError> {
    let mut ops = Vec::with_capacity(2);
    decode_into(inst, pc, cond_passed, &mut ops)?;
    Ok(ops)
}

/// As [`decode`], but appending into a caller-supplied buffer so that no per-instruction allocation happens.
pub fn decode_into(
    inst: u16,
    pc: u32,
    cond_passed: bool,
    ops: &mut Vec<IrOp>,
) -> Result<(), DecodeError> {
    let raw = inst as u32;

    let f = |hi: u32, lo: u32| bitops::bits(raw, hi, lo);

    let hi10 = f(15, 10);
    if hi10 == 0b010000 {
        alu_group(raw, ops)?;
    } else if hi10 == 0b010001 {
        hi_reg_group(raw, pc, ops)?;
    } else if f(15, 11) == 0b01001 {
        // LDR Rd, [PC, #imm8*4]
        let rd = f(10, 8) as u8;
        let offset = f(7, 0) * 4;
        ops.push(IrOp::Load {
            dst: rd,
            // Reading PC in Thumb gives the instruction address plus 4 with bit 1 cleared
            address: Address::Absolute((pc.wrapping_add(4) & !3).wrapping_add(offset)),
            size: MemSize::U32,
            signed: false,
            pre_index: true,
            writeback: false,
        });
        if rd == 15 {
            ops.push(IrOp::JumpExchange {
                target: Value::Reg(15),
            });
        }
    } else if f(15, 11) <= 0b00010 {
        shift_immediate(raw, ops);
    } else if f(15, 11) == 0b00011 {
        add_sub_register(raw, ops);
    } else {
        match f(15, 12) {
            0b0101 => load_store_register(raw, ops)?,
            0b0110 | 0b0111 => load_store_imm5(raw, ops),
            0b1000 => load_store_halfword(raw, ops),
            0b1001 => load_store_sp_relative(raw, ops),
            0b1010 => add_pc_sp(raw, pc, ops),
            0b1011 => misc(raw, ops)?,
            0b1100 => ldm_stm(raw, ops),
            0b1101 => conditional_branch(raw, pc, cond_passed, ops)?,
            0b1111 => {
                return Err(DecodeError::Unsupported {
                    encoding: raw,
                    reason: "32-bit Thumb (BL/BLX) requires instruction pairing",
                });
            }
            0b0010 | 0b0011 => mov_cmp_add_sub_imm(raw, ops),
            _ => {
                if f(15, 11) == 0b11100 {
                    let imm11 = f(10, 0);
                    let offset = sign_extend(imm11, 11) << 1;
                    ops.push(IrOp::Jump {
                        target: Value::Imm(pc.wrapping_add(4).wrapping_add(offset)),
                    });
                } else {
                    return Err(DecodeError::Undefined(raw));
                }
            }
        }
    }

    Ok(())
}

fn add_sub_register(raw: u32, ops: &mut Vec<IrOp>) {
    // Format 2: 00011 I op Rn/imm3 Rs Rd, meaning Rd = Rs +/- Rn
    let immediate = bitops::bit(raw, 10) == 1;
    let sub = bitops::bit(raw, 9) == 1;
    let rn_or_imm = bitops::bits(raw, 8, 6);
    let rs = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;

    ops.push(IrOp::Alu {
        op: if sub { AluOp::Sub } else { AluOp::Add },
        dst: Some(rd),
        a: Value::Reg(rs),
        b: Some(if immediate {
            Value::Imm(rn_or_imm)
        } else {
            Value::Reg(rn_or_imm as u8)
        }),
        set_flags: true,
    });
}

fn shift_immediate(raw: u32, ops: &mut Vec<IrOp>) {
    let kind = bitops::bits(raw, 12, 11);
    let shift = match kind {
        0 => ShiftType::Lsl,
        1 => ShiftType::Lsr,
        _ => ShiftType::Asr,
    };
    let imm5 = bitops::bits(raw, 10, 6);
    let rs = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;
    // For LSR and ASR an imm5 of 0 means a shift of 32
    let amount = if imm5 == 0 && shift != ShiftType::Lsl {
        32
    } else {
        imm5
    };
    ops.push(IrOp::Alu {
        op: AluOp::Mov,
        dst: Some(rd),
        a: Value::Shifted {
            reg: rs,
            shift,
            by: Shifter::Imm(amount as u8),
        },
        b: None,
        set_flags: true,
    });
}

fn mov_cmp_add_sub_imm(raw: u32, ops: &mut Vec<IrOp>) {
    let op = bitops::bits(raw, 12, 11);
    let rd = bitops::bits(raw, 10, 8) as u8;
    let imm = Value::Imm(bitops::bits(raw, 7, 0));
    match op {
        0 => ops.push(IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(rd),
            a: imm,
            b: None,
            set_flags: true,
        }),
        1 => ops.push(IrOp::Alu {
            op: AluOp::Cmp,
            dst: None,
            a: Value::Reg(rd),
            b: Some(imm),
            set_flags: true,
        }),
        _ => ops.push(IrOp::Alu {
            op: if op == 2 { AluOp::Add } else { AluOp::Sub },
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(imm),
            set_flags: true,
        }),
    }
}

fn alu_group(raw: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let op = bitops::bits(raw, 9, 6);
    let rs = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;

    let shift_by_reg = |shift: ShiftType| IrOp::Alu {
        op: AluOp::Mov,
        dst: Some(rd),
        a: Value::Shifted {
            reg: rd,
            shift,
            by: Shifter::Reg(rs),
        },
        b: None,
        set_flags: true,
    };

    let op = match op {
        0 => IrOp::Alu {
            op: AluOp::And,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        1 => IrOp::Alu {
            op: AluOp::Eor,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        2 => shift_by_reg(ShiftType::Lsl),
        3 => shift_by_reg(ShiftType::Lsr),
        4 => shift_by_reg(ShiftType::Asr),
        5 => IrOp::Alu {
            op: AluOp::Adc,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        6 => IrOp::Alu {
            op: AluOp::Sbc,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        7 => shift_by_reg(ShiftType::Ror),
        8 => IrOp::Alu {
            op: AluOp::Tst,
            dst: None,
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        // NEG Rd, Rs == RSB Rd, Rs, #0
        9 => IrOp::Alu {
            op: AluOp::Rsb,
            dst: Some(rd),
            a: Value::Reg(rs),
            b: Some(Value::Imm(0)),
            set_flags: true,
        },
        10 => IrOp::Alu {
            op: AluOp::Cmp,
            dst: None,
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        11 => IrOp::Alu {
            op: AluOp::Cmn,
            dst: None,
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        12 => IrOp::Alu {
            op: AluOp::Orr,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        13 => IrOp::Mul {
            dst: rd,
            a: rd,
            b: rs,
            accumulate: None,
            set_flags: true,
        },
        14 => IrOp::Alu {
            op: AluOp::Bic,
            dst: Some(rd),
            a: Value::Reg(rd),
            b: Some(Value::Reg(rs)),
            set_flags: true,
        },
        _ => IrOp::Alu {
            op: AluOp::Mvn,
            dst: Some(rd),
            a: Value::Reg(rs),
            b: None,
            set_flags: true,
        },
    };
    ops.push(op);
    Ok(())
}

fn hi_reg_group(raw: u32, pc: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    // The high-register group: op(2) H1 H2 Rs(3) Rd(3), with Rs's high bit in bit6
    let op = bitops::bits(raw, 9, 8);
    let rs = bitops::bits(raw, 6, 3) as u8;
    let rd = (bitops::bits(raw, 2, 0) | (bitops::bit(raw, 7) << 3)) as u8;

    // A PC source is folded into an immediate at translation time, for the same reason the A32
    // decoder folds it: r15 does not advance inside a block.
    //
    // The value is `instr + 4` **without** clearing bit 1. That is not the alignment `LDR Rd,[pc,#…]`
    // and `ADD Rd,PC,#imm` use — those both word-align their PC base and have their own arms — and
    // it is not academic: code that folds a PC-relative literal into a pointer just below the
    // module base needs the unaligned value to wrap to `module_base - 4`. Reading r15 raw instead
    // yields an address four bytes higher and faults on the following access.
    let rs_value = if rs == 15 {
        Value::Imm(pc.wrapping_add(4))
    } else {
        Value::Reg(rs)
    };

    match op {
        0 => ops.push(IrOp::Alu {
            op: AluOp::Add,
            dst: Some(rd),
            // The destination is an operand of `add rd, rs`, so `add pc, rs` has to read the same
            // biased PC the source does rather than whatever r15 happens to hold in a block.
            a: if rd == 15 {
                Value::Imm(pc.wrapping_add(4))
            } else {
                Value::Reg(rd)
            },
            b: Some(rs_value),
            set_flags: false,
        }),
        1 => ops.push(IrOp::Alu {
            op: AluOp::Cmp,
            dst: None,
            a: Value::Reg(rd),
            b: Some(rs_value),
            set_flags: true,
        }),
        2 => ops.push(IrOp::Alu {
            op: AluOp::Mov,
            dst: Some(rd),
            a: rs_value,
            b: None,
            set_flags: false,
        }),
        _ => {
            // BX / BLX (register): bit 7 selects which. BLX also writes LR with the return
            // address — the next halfword, its Thumb bit set — exactly as the A32 decoder does
            // for `blx rm`. Leaving it out made the callee return to the *caller's* return
            // address: its own frame was never popped, and a later frame popped zero into PC.
            if bitops::bit(raw, 7) == 1 {
                ops.push(IrOp::Alu {
                    op: AluOp::Mov,
                    dst: Some(14),
                    a: Value::Imm(pc.wrapping_add(2) | 1),
                    b: None,
                    set_flags: false,
                });
            }
            ops.push(IrOp::JumpExchange { target: rs_value });
        }
    }

    // `add pc, rs` and `mov pc, rs` are branches, not arithmetic with a register for a destination:
    // the write *is* the jump. Leaving it as a plain write to r15 left the block's control flow
    // alone, so the driver fell through to the next instruction -- on `w52ca_brain-challenge` that
    // turned every `add pc, r3` jump-table dispatch into "run the instruction after the table",
    // which skipped a whole initialisation path and faulted 600 000 instructions later at address 0.
    //
    // `Jump`, not `JumpExchange`: in Thumb these two do not switch instruction sets (that is BX/BLX,
    // op 3 above), so bit 0 of the target is not a state selector here.
    if matches!(op, 0 | 2) && rd == 15 {
        ops.push(IrOp::Jump {
            target: Value::Reg(15),
        });
    }
    Ok(())
}

fn load_store_register(raw: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    let op = bitops::bits(raw, 11, 9);
    let rm = bitops::bits(raw, 8, 6) as u8;
    let rn = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;

    let address = Address::Register {
        base: rn,
        offset: Value::Reg(rm),
        add: true,
    };

    let (size, signed, load) = match op {
        0 => (MemSize::U32, false, false),
        1 => (MemSize::U16, false, false),
        2 => (MemSize::U8, false, false),
        3 => (MemSize::U8, true, true),
        4 => (MemSize::U32, false, true),
        5 => (MemSize::U16, false, true),
        6 => (MemSize::U8, false, true),
        _ => (MemSize::U16, true, true),
    };

    if load {
        ops.push(IrOp::Load {
            dst: rd,
            address,
            size,
            signed,
            pre_index: true,
            writeback: false,
        });
    } else {
        ops.push(IrOp::Store {
            src: rd,
            address,
            size,
            pre_index: true,
            writeback: false,
        });
    }
    Ok(())
}

fn load_store_imm5(raw: u32, ops: &mut Vec<IrOp>) {
    let load = bitops::bit(raw, 11) == 1;
    let byte = bitops::bit(raw, 12) == 1;
    let imm5 = bitops::bits(raw, 10, 6);
    let rn = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;

    let size = if byte { MemSize::U8 } else { MemSize::U32 };
    let offset = Value::Imm(imm5 * size.bytes());
    let address = Address::Register {
        base: rn,
        offset,
        add: true,
    };

    if load {
        ops.push(IrOp::Load {
            dst: rd,
            address,
            size,
            signed: false,
            pre_index: true,
            writeback: false,
        });
    } else {
        ops.push(IrOp::Store {
            src: rd,
            address,
            size,
            pre_index: true,
            writeback: false,
        });
    }
}

fn load_store_halfword(raw: u32, ops: &mut Vec<IrOp>) {
    let load = bitops::bit(raw, 11) == 1;
    let imm5 = bitops::bits(raw, 10, 6);
    let rn = bitops::bits(raw, 5, 3) as u8;
    let rd = bitops::bits(raw, 2, 0) as u8;

    let address = Address::Register {
        base: rn,
        offset: Value::Imm(imm5 * 2),
        add: true,
    };

    if load {
        ops.push(IrOp::Load {
            dst: rd,
            address,
            size: MemSize::U16,
            signed: false,
            pre_index: true,
            writeback: false,
        });
    } else {
        ops.push(IrOp::Store {
            src: rd,
            address,
            size: MemSize::U16,
            pre_index: true,
            writeback: false,
        });
    }
}

fn load_store_sp_relative(raw: u32, ops: &mut Vec<IrOp>) {
    let load = bitops::bit(raw, 11) == 1;
    let rd = bitops::bits(raw, 10, 8) as u8;
    let offset = bitops::bits(raw, 7, 0) * 4;

    let address = Address::Register {
        base: 13,
        offset: Value::Imm(offset),
        add: true,
    };

    if load {
        ops.push(IrOp::Load {
            dst: rd,
            address,
            size: MemSize::U32,
            signed: false,
            pre_index: true,
            writeback: false,
        });
    } else {
        ops.push(IrOp::Store {
            src: rd,
            address,
            size: MemSize::U32,
            pre_index: true,
            writeback: false,
        });
    }
}

fn add_pc_sp(raw: u32, pc: u32, ops: &mut Vec<IrOp>) {
    let use_sp = bitops::bit(raw, 11) == 1;
    let rd = bitops::bits(raw, 10, 8) as u8;
    let imm = bitops::bits(raw, 7, 0) * 4;
    // PC-relative: reading PC in Thumb gives the instruction address plus 4 with bit 1 cleared, folded at translation time
    let source = if use_sp {
        Value::Reg(13)
    } else {
        Value::Imm(pc.wrapping_add(4) & !3)
    };
    ops.push(IrOp::Alu {
        op: AluOp::Add,
        dst: Some(rd),
        a: source,
        b: Some(Value::Imm(imm)),
        set_flags: false,
    });
}

fn misc(raw: u32, ops: &mut Vec<IrOp>) -> Result<(), DecodeError> {
    match bitops::bits(raw, 11, 8) {
        // `1011 0000 0 imm7` / `1011 0000 1 imm7`: ADD/SUB SP.
        //
        // Only `0b0000`. `1011 0001` is `CBZ`/`CBNZ` (`1011 0z01 iiiii nnn`), which used to be
        // answered here as well — so a `CBZ` moved the stack pointer by whatever `bits[6:0]` happened
        // to say instead of branching. Silent, plausible, and exactly the shape this decoder exists to
        // refuse; it fell out of the coverage histogram asking "which top bytes are unimplemented",
        // where `0xB1xx` was conspicuously absent.
        0b0000 => {
            let offset = bitops::bits(raw, 6, 0) * 4;
            let sub = bitops::bit(raw, 7) == 1;
            ops.push(IrOp::Alu {
                op: if sub { AluOp::Sub } else { AluOp::Add },
                dst: Some(13),
                a: Value::Reg(13),
                b: Some(Value::Imm(offset)),
                set_flags: false,
            });
            Ok(())
        }
        0b0100 | 0b0101 => {
            // PUSH {list} [, LR]
            let mut regs = bitops::bits(raw, 7, 0) as u16;
            if bitops::bit(raw, 8) == 1 {
                regs |= 1 << 14;
            }
            ops.push(IrOp::StoreMultiple {
                base: 13,
                regs,
                add: false,
                pre_index: true,
                writeback: true,
            });
            Ok(())
        }
        0b1100 | 0b1101 => {
            // POP {list} [, PC]
            let mut regs = bitops::bits(raw, 7, 0) as u16;
            if bitops::bit(raw, 8) == 1 {
                regs |= 1 << 15;
            }
            ops.push(IrOp::LoadMultiple {
                base: 13,
                regs,
                add: true,
                pre_index: false,
                writeback: true,
            });
            if regs & (1 << 15) != 0 {
                ops.push(IrOp::JumpExchange {
                    target: Value::Reg(15),
                });
            }
            Ok(())
        }
        // SXTH / SXTB / UXTH / UXTB (ARMv6): `1011 0010 ss ww`, where bits 7-6 pick the pair and
        // the width. A pack built for an ARMv6 handset executes these on its start-up path, and
        // the ARM decoder already expresses the same four with two shifts — shared here so the
        // two instruction sets cannot drift into different answers.
        0b0010 => {
            let rm = bitops::bits(raw, 5, 3) as u8;
            let rd = bitops::bits(raw, 2, 0) as u8;
            let (signed, bits) = match bitops::bits(raw, 7, 6) {
                0 => (true, 16),
                1 => (true, 8),
                2 => (false, 16),
                _ => (false, 8),
            };
            super::a32::sign_or_zero_extend(rd, rm, signed, bits, ops);
            Ok(())
        }
        // BKPT: 1011 1110 imm8
        0b1110 => {
            ops.push(IrOp::Breakpoint {
                imm: bitops::bits(raw, 7, 0),
            });
            Ok(())
        }
        _ => Err(DecodeError::Unsupported {
            encoding: raw,
            reason: "Thumb misc: REV / CBZ / BKPT",
        }),
    }
}

fn ldm_stm(raw: u32, ops: &mut Vec<IrOp>) {
    let load = bitops::bit(raw, 11) == 1;
    let base = bitops::bits(raw, 10, 8) as u8;
    let regs = bitops::bits(raw, 7, 0) as u16;

    if load {
        let writeback = regs & (1 << base) == 0;
        ops.push(IrOp::LoadMultiple {
            base,
            regs,
            add: true,
            pre_index: false,
            writeback,
        });
    } else {
        ops.push(IrOp::StoreMultiple {
            base,
            regs,
            add: true,
            pre_index: false,
            writeback: true,
        });
    }
}

fn conditional_branch(
    raw: u32,
    pc: u32,
    cond_passed: bool,
    ops: &mut Vec<IrOp>,
) -> Result<(), DecodeError> {
    let cond = bitops::bits(raw, 11, 8);
    if cond == 0xF {
        ops.push(IrOp::Swi {
            imm: bitops::bits(raw, 7, 0),
        });
        return Ok(());
    }
    // 1101 1110 is the permanently undefined instruction (not an unconditional branch)
    if cond == 0xE {
        return Err(DecodeError::Undefined(raw));
    }
    if cond_passed {
        let offset = sign_extend(bitops::bits(raw, 7, 0), 8) << 1;
        ops.push(IrOp::Jump {
            target: Value::Imm(pc.wrapping_add(4).wrapping_add(offset)),
        });
    }
    Ok(())
}

/// Joins the two halfwords of `BL`/`BLX` (the first is `upper`, the second `lower`).
///
/// `pc` is the address of the **first** halfword; the return address goes to LR with the Thumb bit set.
pub fn decode_bl_pair(upper: u16, lower: u16, pc: u32) -> Result<Vec<IrOp>, DecodeError> {
    let upper = upper as u32;
    let lower = lower as u32;

    if bitops::bits(upper, 15, 11) != 0b11110 {
        return Err(DecodeError::Undefined(upper << 16 | lower));
    }

    let s = bitops::bit(upper, 10);
    let j1 = bitops::bit(lower, 13);
    let j2 = bitops::bit(lower, 11);
    let i1 = u32::from(j1 ^ s == 0);
    let i2 = u32::from(j2 ^ s == 0);

    let imm10 = bitops::bits(upper, 9, 0);
    let imm11 = bitops::bits(lower, 10, 0);
    let imm32 = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
    let offset = sign_extend(imm32, 25);

    let target = pc.wrapping_add(4).wrapping_add(offset);
    let link_register = pc.wrapping_add(4) | 1;

    let mut ops = Vec::with_capacity(2);
    ops.push(IrOp::Alu {
        op: AluOp::Mov,
        dst: Some(14),
        a: Value::Imm(link_register),
        b: None,
        set_flags: false,
    });

    if bitops::bit(lower, 12) == 1 {
        // BL: stays in Thumb
        ops.push(IrOp::JumpExchange {
            target: Value::Imm(target | 1),
        });
    } else {
        // BLX (imm): switches to ARM state
        ops.push(IrOp::JumpExchange {
            target: Value::Imm(target & !3),
        });
    }

    Ok(ops)
}

/// Whether this is the first halfword of a 32-bit Thumb instruction (a `BL`/`BLX` prefix).
pub fn is_bl_prefix(inst: u16) -> bool {
    let raw = inst as u32;
    bitops::bits(raw, 15, 11) == 0b11110
}

/// Whether this is the second halfword of `BL`/`BLX`.
pub fn is_bl_suffix(inst: u16) -> bool {
    let raw = inst as u32;
    bitops::bits(raw, 15, 14) == 0b11 && bitops::bit(raw, 12) <= 1
}

/// Whether the encoding is `B<cond>` (0xDE is UDF and 0xDF is SWI, neither of which counts).
pub fn is_conditional_branch(inst: u16) -> bool {
    let raw = inst as u32;
    let cond = bitops::bits(raw, 11, 8);
    bitops::bits(raw, 15, 12) == 0b1101 && cond != 0xF && cond != 0xE
}

/// Decodes `B<cond>`; the condition is evaluated at run time.
pub fn decode_conditional_branch(inst: u16, pc: u32, ops: &mut Vec<IrOp>) {
    let raw = inst as u32;
    let cond = bitops::bits(raw, 11, 8);
    let offset = sign_extend(bitops::bits(raw, 7, 0), 8) << 1;
    ops.push(IrOp::BranchConditional {
        cond,
        target: Value::Imm(pc.wrapping_add(4).wrapping_add(offset)),
    });
}
