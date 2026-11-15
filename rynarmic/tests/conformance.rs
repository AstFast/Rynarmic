//! A decoding conformance table: for each implemented instruction family, an encoding and the IR shape it must produce.
//!
//! Why it exists: the four defects that appeared earlier were all **identity mix-ups** —
//! `MUL` executed as `AND`, `CMP r1,#0` taken for `MSR imm`, `BXJ` taken for `BLX`,
//! and Thumb `BX lr` decoded as `BX r6`. None of those raises an error; they just compute the wrong answer, which makes them hard to pin down
//! by "run a program and look at the result". With this table, any damage to the dispatch order fails immediately and by name.

use rynarmic::ir::{Address, AluOp, IrOp, MemSize, ShiftType, Shifter, Value};
use rynarmic::{DecodeError, block, decoder};

/// The first meaningful IR the decoder is expected to produce.
#[derive(Debug, Clone)]
enum Expect {
    Alu(AluOp),
    Mul,
    MulLong,
    MulHalf,
    Saturating,
    Clz,
    Swap,
    Load {
        size: MemSize,
        signed: bool,
    },
    Store {
        size: MemSize,
    },
    LoadDouble,
    StoreDouble,
    LoadMultiple,
    StoreMultiple,
    /// B: a direct jump
    Jump,
    /// BL: write LR (a Mov) first, then jump
    JumpLink,
    /// BX / BXJ
    Exchange,
    /// BLX(reg): write LR, then jump
    ExchangeLink,
    ReadStatus,
    WriteStatus,
    Swi,
    Breakpoint,
    Nop,
}

fn matches(op: &IrOp, expect: &Expect) -> bool {
    match (expect, op) {
        (Expect::Alu(want), IrOp::Alu { op, .. }) => op == want,
        (Expect::Mul, IrOp::Mul { .. }) => true,
        (Expect::MulLong, IrOp::MulLong { .. }) => true,
        (Expect::MulHalf, IrOp::MulHalf { .. }) => true,
        (Expect::Saturating, IrOp::Saturating { .. }) => true,
        (Expect::Clz, IrOp::Clz { .. }) => true,
        (Expect::Swap, IrOp::Swap { .. }) => true,
        (
            Expect::Load { size, signed },
            IrOp::Load {
                size: s, signed: g, ..
            },
        ) => s == size && g == signed,
        (Expect::Store { size }, IrOp::Store { size: s, .. }) => s == size,
        (Expect::LoadDouble, IrOp::LoadDouble { .. }) => true,
        (Expect::StoreDouble, IrOp::StoreDouble { .. }) => true,
        (Expect::LoadMultiple, IrOp::LoadMultiple { .. }) => true,
        (Expect::StoreMultiple, IrOp::StoreMultiple { .. }) => true,
        (Expect::Jump, IrOp::Jump { .. }) => true,
        (Expect::Exchange, IrOp::JumpExchange { .. }) => true,
        (Expect::ReadStatus, IrOp::ReadStatus { .. }) => true,
        (Expect::WriteStatus, IrOp::WriteStatus { .. }) => true,
        (Expect::Swi, IrOp::Swi { .. }) => true,
        (Expect::Breakpoint, IrOp::Breakpoint { .. }) => true,
        (Expect::Nop, IrOp::Nop) => true,
        _ => false,
    }
}

fn check_arm(inst: u32, expect: Expect, label: &str) {
    let ops = decoder::a32::decode(inst, 0x1000, true)
        .unwrap_or_else(|e| panic!("{label}: decoding {inst:#010x} failed: {e}"));

    let ok = match expect {
        // BL's first operation writes LR
        Expect::JumpLink => {
            ops.len() == 2
                && matches!(
                    ops[0],
                    IrOp::Alu {
                        op: AluOp::Mov,
                        dst: Some(14),
                        ..
                    }
                )
                && matches!(ops[1], IrOp::Jump { .. })
        }
        Expect::ExchangeLink => {
            ops.len() == 2
                && matches!(
                    ops[0],
                    IrOp::Alu {
                        op: AluOp::Mov,
                        dst: Some(14),
                        ..
                    }
                )
                && matches!(ops[1], IrOp::JumpExchange { .. })
        }
        ref want => ops.first().is_some_and(|op| matches(op, want)),
    };
    assert!(
        ok,
        "{label}: {inst:#010x} decoded to {ops:?}, expected {expect:?}"
    );
}

fn check_thumb(inst: u16, expect: Expect, label: &str) {
    let ops = decoder::thumb::decode(inst, 0x1000, true)
        .unwrap_or_else(|e| panic!("{label}: decoding {inst:#06x} failed: {e}"));
    assert!(
        ops.first().is_some_and(|op| matches(op, &expect)),
        "{label}: {inst:#06x} decoded to {ops:?}, expected {expect:?}"
    );
}

fn check_undefined_arm(inst: u32) {
    assert!(
        matches!(
            decoder::a32::decode(inst, 0x1000, true),
            Err(DecodeError::Undefined(_)) | Err(DecodeError::Unsupported { .. })
        ),
        "{inst:#010x} should have been rejected"
    );
}

/// Takes the `Alu`'s second operand (`MOV`/`MVN` use the first).
fn alu_operand(inst: u32) -> Value {
    let ops = decoder::a32::decode(inst, 0x1000, true)
        .unwrap_or_else(|e| panic!("{inst:#010x}: decoding failed: {e}"));
    match ops.first() {
        Some(IrOp::Alu { a, b, .. }) => (*b).unwrap_or(*a),
        other => panic!("{inst:#010x}: expected an Alu, got {other:?}"),
    }
}

#[test]
fn a32_data_processing_families() {
    check_arm(0xE3A0_0005, Expect::Alu(AluOp::Mov), "mov imm");
    check_arm(0xE1A0_0001, Expect::Alu(AluOp::Mov), "mov reg");
    check_arm(0xE041_0002, Expect::Alu(AluOp::Sub), "sub reg");
    check_arm(0xE350_0005, Expect::Alu(AluOp::Cmp), "cmp imm");
    check_arm(
        0xE351_0000,
        Expect::Alu(AluOp::Cmp),
        "cmp #0 (once mistaken for MSR imm)",
    );
    check_arm(0xE311_0002, Expect::Alu(AluOp::Tst), "tst reg");
    check_arm(0xE3E0_0000, Expect::Alu(AluOp::Mvn), "mvn imm");
    check_arm(0xE2A1_0001, Expect::Alu(AluOp::Adc), "adc");
    check_arm(0xE1A0_0231, Expect::Alu(AluOp::Mov), "mov reg, lsr reg");
    check_arm(0xE1A0_0251, Expect::Alu(AluOp::Mov), "mov reg, asr reg");
    check_arm(0xE1A0_0271, Expect::Alu(AluOp::Mov), "mov reg, ror reg");
}

#[test]
fn a32_shift_zero_and_rrx_shapes() {
    use ShiftType::{Asr, Lsl, Lsr, Ror, Rrx};
    use Shifter::{Imm, Reg};

    assert_eq!(
        alu_operand(0xE1A0_0231),
        Value::Shifted {
            reg: 1,
            shift: Lsr,
            by: Reg(2),
        }
    );
    assert_eq!(
        alu_operand(0xE1A0_0251),
        Value::Shifted {
            reg: 1,
            shift: Asr,
            by: Reg(2),
        }
    );
    assert_eq!(
        alu_operand(0xE1A0_0271),
        Value::Shifted {
            reg: 1,
            shift: Ror,
            by: Reg(2),
        }
    );
    // An immediate of 0: 32 for LSR and ASR, RRX for ROR, and only LSL is a real no-op
    assert_eq!(
        alu_operand(0xE1A0_0021),
        Value::Shifted {
            reg: 1,
            shift: Lsr,
            by: Imm(32),
        }
    );
    assert_eq!(
        alu_operand(0xE1A0_0041),
        Value::Shifted {
            reg: 1,
            shift: Asr,
            by: Imm(32),
        }
    );
    assert_eq!(
        alu_operand(0xE1A0_0061),
        Value::Shifted {
            reg: 1,
            shift: Rrx,
            by: Imm(0),
        }
    );
    assert_eq!(
        alu_operand(0xE1A0_0001),
        Value::Shifted {
            reg: 1,
            shift: Lsl,
            by: Imm(0),
        }
    );
}

fn load_offset(inst: u32) -> Value {
    let ops = decoder::a32::decode(inst, 0x1000, true)
        .unwrap_or_else(|e| panic!("{inst:#010x}: decoding failed: {e}"));
    match ops.first() {
        Some(IrOp::Load {
            address: Address::Register { offset, .. },
            ..
        }) => *offset,
        other => panic!("{inst:#010x}: expected a Load with a register offset, got {other:?}"),
    }
}

#[test]
fn a32_memory_register_shift_shapes() {
    use ShiftType::{Asr, Lsr, Ror, Rrx};
    use Shifter::Imm;

    // ldr r0, [r1, r2, <type> <amount>]
    assert_eq!(
        load_offset(0xE791_0122),
        Value::Shifted {
            reg: 2,
            shift: Lsr,
            by: Imm(2),
        }
    );
    assert_eq!(
        load_offset(0xE791_0022),
        Value::Shifted {
            reg: 2,
            shift: Lsr,
            by: Imm(32),
        }
    );
    assert_eq!(
        load_offset(0xE791_0062),
        Value::Shifted {
            reg: 2,
            shift: Rrx,
            by: Imm(0),
        }
    );
    assert_eq!(
        load_offset(0xE791_0142),
        Value::Shifted {
            reg: 2,
            shift: Asr,
            by: Imm(2),
        }
    );
    assert_eq!(
        load_offset(0xE791_0162),
        Value::Shifted {
            reg: 2,
            shift: Ror,
            by: Imm(2),
        }
    );
}

#[test]
fn a32_multiply_families() {
    check_arm(0xE000_0192, Expect::Mul, "mul");
    check_arm(0xE020_3192, Expect::Mul, "mla");
    check_arm(0xE081_0293, Expect::MulLong, "umull");
    check_arm(0xE0C1_0293, Expect::MulLong, "smull");
    check_arm(0xE140_2384, Expect::MulHalf, "smlalxy");
    check_arm(0xE102_0051, Expect::Saturating, "qadd");
    check_arm(0xE16F_0F11, Expect::Clz, "clz");
}

#[test]
fn a32_memory_families() {
    check_arm(
        0xE590_0000,
        Expect::Load {
            size: MemSize::U32,
            signed: false,
        },
        "ldr",
    );
    check_arm(
        0xE5D0_0000,
        Expect::Load {
            size: MemSize::U8,
            signed: false,
        },
        "ldrb",
    );
    check_arm(0xE5C0_1000, Expect::Store { size: MemSize::U8 }, "strb");
    check_arm(
        0xE1D1_00B4,
        Expect::Load {
            size: MemSize::U16,
            signed: false,
        },
        "ldrh",
    );
    check_arm(
        0xE1D3_10D0,
        Expect::Load {
            size: MemSize::U8,
            signed: true,
        },
        "ldrsb",
    );
    check_arm(
        0xE1D3_20F2,
        Expect::Load {
            size: MemSize::U16,
            signed: true,
        },
        "ldrsh",
    );
    check_arm(0xE1C1_00D8, Expect::LoadDouble, "ldrd");
    check_arm(0xE1C1_00F8, Expect::StoreDouble, "strd");
    check_arm(0xE102_0091, Expect::Swap, "swp");
    check_arm(0xE8A9_000E, Expect::StoreMultiple, "stmia");
    check_arm(0xE890_0070, Expect::LoadMultiple, "ldmia");
}

#[test]
fn a32_control_flow_families() {
    check_arm(0xEAFF_FFFE, Expect::Jump, "b");
    check_arm(0xEB00_0000, Expect::JumpLink, "bl");
    check_arm(0xE12F_FF1E, Expect::Exchange, "bx lr");
    check_arm(0xE12F_FF3E, Expect::ExchangeLink, "blx lr");
    check_arm(0xE12F_FF2E, Expect::Exchange, "bxj lr (v5TEJ)");
    check_arm(
        0x1A00_0001,
        Expect::Jump,
        "bne (the single-step path runs with the condition already decided)",
    );
    check_arm(
        0x0A00_0001,
        Expect::Jump,
        "beq (the single-step path runs with the condition already decided)",
    );
}

#[test]
fn a32_system_and_misc_families() {
    check_arm(0xE10F_0000, Expect::ReadStatus, "mrs r0, cpsr");
    check_arm(0xE321_F013, Expect::WriteStatus, "msr cpsr_c, #0x13");
    check_arm(0xEF00_0000, Expect::Swi, "swi");
    check_arm(0xE120_0070, Expect::Breakpoint, "bkpt");
    check_arm(0xF57F_F004, Expect::Nop, "pld (the unconditional space)");
}

#[test]
fn a32_rejects_out_of_scope_encodings() {
    check_undefined_arm(0xEE00_0A10); // vmov s0, r0 -- VFP is not implemented
    check_undefined_arm(0xEC00_0000); // ldc -- the coprocessor space
    check_undefined_arm(0xF000_0000); // an ARMv7 instruction in the unconditional space
    check_undefined_arm(0xE1C1_10D8); // ldrd with an odd Rt
    check_undefined_arm(0xE140_1384); // smlalxy with an odd RdLo
}

#[test]
fn thumb_format_families() {
    check_thumb(0x2000, Expect::Alu(AluOp::Mov), "movs imm8");
    check_thumb(0x3901, Expect::Alu(AluOp::Sub), "subs imm8");
    check_thumb(0x1808, Expect::Alu(AluOp::Add), "adds, register form");
    check_thumb(
        0x4288,
        Expect::Alu(AluOp::Cmp),
        "cmp in the high-register group",
    );
    check_thumb(0x4008, Expect::Alu(AluOp::And), "and (the ALU group)");
    check_thumb(0x4358, Expect::Mul, "mul (the ALU group)");
    check_thumb(0x4770, Expect::Exchange, "bx lr");
    check_thumb(
        0x4801,
        Expect::Load {
            size: MemSize::U32,
            signed: false,
        },
        "ldr, literal form",
    );
    check_thumb(
        0x6801,
        Expect::Load {
            size: MemSize::U32,
            signed: false,
        },
        "ldr, immediate form",
    );
    check_thumb(
        0x7801,
        Expect::Load {
            size: MemSize::U8,
            signed: false,
        },
        "ldrb",
    );
    check_thumb(
        0x8801,
        Expect::Load {
            size: MemSize::U16,
            signed: false,
        },
        "ldrh",
    );
    check_thumb(
        0x5E41,
        Expect::Load {
            size: MemSize::U16,
            signed: true,
        },
        "ldrsh",
    );
    check_thumb(
        0x9801,
        Expect::Load {
            size: MemSize::U32,
            signed: false,
        },
        "ldr, SP-relative",
    );
    check_thumb(0xB580, Expect::StoreMultiple, "push");
    check_thumb(0xBD80, Expect::LoadMultiple, "pop (with pc)");
    check_thumb(0xC808, Expect::LoadMultiple, "ldmia");
    check_thumb(
        0xD1FC,
        Expect::Jump,
        "bne (the single-step path runs with the condition already decided)",
    );
    check_thumb(0xE7FE, Expect::Jump, "b");
    check_thumb(0xDF00, Expect::Swi, "swi");
    check_thumb(0xBE00, Expect::Breakpoint, "bkpt");
}

#[test]
fn block_translation_uses_runtime_conditional_branch() {
    // This is the crux of block-cache correctness: a block runs repeatedly, so the condition must not be fixed at translation time
    let mut memory = rynarmic::memory::SimpleMemory::new(0, 0x2000);
    memory.write_words(0x1000, &[0x1A00_0001]);

    let block = match block::translate(&mut memory, 0x1000, false).unwrap() {
        block::Translation::Block(b) => b,
        other => panic!("expected a basic block, got {other:?}"),
    };
    assert!(
        matches!(block.ops[0], IrOp::BranchConditional { cond: 0x1, .. }),
        "a conditional branch inside a block has to be decided at run time: {:?}",
        block.ops
    );

    let mut memory = rynarmic::memory::SimpleMemory::new(0, 0x2000);
    memory.write_thumb(0x1000, &[0xD1FC]);
    let block = match block::translate(&mut memory, 0x1000, true).unwrap() {
        block::Translation::Block(b) => b,
        other => panic!("expected a basic block, got {other:?}"),
    };
    assert!(matches!(
        block.ops[0],
        IrOp::BranchConditional { cond: 0x1, .. }
    ));
}

#[test]
fn thumb_32bit_pair_families() {
    let bl = decoder::thumb::decode_bl_pair(0xF000, 0xF806, 0x1000).unwrap();
    assert!(
        matches!(
            bl[0],
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(14),
                ..
            }
        ) && matches!(bl[1], IrOp::JumpExchange { .. }),
        "BL should write LR before jumping: {bl:?}"
    );

    let blx = decoder::thumb::decode_bl_pair(0xF000, 0xE806, 0x1000).unwrap();
    assert!(
        matches!(
            blx[0],
            IrOp::Alu {
                op: AluOp::Mov,
                dst: Some(14),
                ..
            }
        ) && matches!(blx[1], IrOp::JumpExchange { .. }),
        "BLX should write LR before switching state: {blx:?}"
    );
}

#[test]
fn thumb_rejects_undefined_encodings() {
    assert!(
        decoder::thumb::decode(0xDE00, 0x1000, true).is_err(),
        "0xDE is UDF"
    );
    assert!(
        decoder::thumb::decode(0xF000, 0x1000, true).is_err(),
        "a BL prefix cannot stand as an instruction of its own"
    );
}

#[test]
fn translation_shapes_are_stable() {
    // A translation has to put the terminal instruction last, with next_pc pointing at the sequential successor
    let mut memory = rynarmic::memory::SimpleMemory::new(0, 0x2000);
    memory.write_words(0x1000, &[0xE3A0_0001, 0xEA00_0000]);

    let block = match block::translate(&mut memory, 0x1000, false).unwrap() {
        block::Translation::Block(b) => b,
        other => panic!("expected a basic block, got {other:?}"),
    };
    assert_eq!(block.instruction_count, 2);
    assert!(matches!(block.ops.last(), Some(IrOp::Jump { .. })));
    assert_eq!(block.terminal_pc, 0x1004);
}
