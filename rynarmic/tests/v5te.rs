//! Vector tests for the instructions ARMv5TE and ARMv5TEJ added.

use rynarmic::{Cpsr, Cpu, StepOutcome, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;

fn cpu_with(words: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu
}

#[test]
fn mul_and_mla() {
    let mut cpu = cpu_with(&[
        0xE000_0192, // mul r0, r1, r2
        0xE020_3192, // mla r0, r1, r2, r3
    ]);
    cpu.state.set_reg(1, 6);
    cpu.state.set_reg(2, 7);
    cpu.state.set_reg(3, 1);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 42);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 43);
}

#[test]
fn umull_and_smull() {
    let mut cpu = cpu_with(&[
        0xE081_0293, // umull r0, r1, r2, r3
        0xE0C1_0293, // smull r0, r1, r2, r3  (lo=r0, hi=r1)
    ]);
    cpu.state.set_reg(2, 0x1_0000);
    cpu.state.set_reg(3, 0x1_0000);

    cpu.step().unwrap();
    assert_eq!(
        (cpu.state.reg(0), cpu.state.reg(1)),
        (0, 1),
        "0x10000^2 = 2^32"
    );

    // Signed: -2 * 3 = -6
    cpu.state.set_reg(2, (-2i32) as u32);
    cpu.state.set_reg(3, 3);
    cpu.step().unwrap();
    assert_eq!(
        (cpu.state.reg(0), cpu.state.reg(1)),
        (0xFFFF_FFFA, 0xFFFF_FFFF)
    );
}

#[test]
fn clz() {
    let mut cpu = cpu_with(&[0xE16F_0F11]); // clz r0, r1
    cpu.state.set_reg(1, 1);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 31);

    cpu.state.set_reg(1, 0);
    cpu.state.set_pc(CODE);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 32);
}

#[test]
fn halfword_and_signed_transfers() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE1C1_20B4, // strh r2, [r1, #4]
            0xE1D1_00B4, // ldrh r0, [r1, #4]
            0xE1D3_10D0, // ldrsb r1, [r3]
            0xE1D3_20F2, // ldrsh r2, [r3, #2]
        ],
    );
    memory.write_bytes(DATA, &[0x80, 0x00, 0x00, 0x80]);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, DATA);
    cpu.state.set_reg(2, 0xBEAF);
    cpu.state.set_reg(3, DATA);

    cpu.step().unwrap();
    assert_eq!(cpu.memory.peek_u32(DATA + 4) & 0xFFFF, 0xBEAF);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0xBEAF, "LDRH should zero-extend");

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(1), 0xFFFF_FF80, "LDRSB should sign-extend");

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(2), 0xFFFF_8000, "LDRSH should sign-extend");
}

#[test]
fn ldrd_strd_round_trip() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE1C1_00F8, // strd r0, [r1, #8]
            0xE1C1_20D8, // ldrd r2, [r1, #8]
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, 0x1111_1111);
    cpu.state.set_reg(1, DATA);
    cpu.state.set_reg(2, 0);
    cpu.state.set_reg(3, 0);

    cpu.step().unwrap();
    assert_eq!(cpu.memory.peek_u32(DATA + 8), 0x1111_1111);
    assert_eq!(
        cpu.memory.peek_u32(DATA + 12),
        DATA,
        "r0+1 is r1, the base register"
    );

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(2), 0x1111_1111);
    assert_eq!(cpu.state.reg(3), DATA);
}

#[test]
fn swp_exchanges_with_memory() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE102_0091]); // swp r0, r1, [r2]
    memory.write_words(DATA, &[0xAAAA_AAAA]);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, 0x5555_5555);
    cpu.state.set_reg(2, DATA);

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(0),
        0xAAAA_AAAA,
        "SWP should read the old value back into Rd"
    );
    assert_eq!(cpu.memory.peek_u32(DATA), 0x5555_5555);
}

#[test]
fn qadd_saturates_and_sets_q() {
    let mut cpu = cpu_with(&[
        0xE102_0051, // qadd r0, r1, r2   (Rm=r1, Rn=r2)
        0xE102_0051, // qadd again, to check that Q is sticky
    ]);
    cpu.state.set_reg(1, 0x7FFF_FFFF);
    cpu.state.set_reg(2, 1);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0x7FFF_FFFF, "saturates up to INT32_MAX");
    assert!(cpu.state.cpsr.q());

    cpu.state.set_reg(1, 1);
    cpu.state.set_reg(2, 1);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 2);
}

#[test]
fn bkpt_reports_exception() {
    let mut cpu = cpu_with(&[0xE120_0070]);
    let result = cpu.step();
    match result {
        Err(rynarmic::cpu::ExceptionKind::Breakpoint(0)) => {}
        other => panic!("expected Breakpoint(0), got {other:?}"),
    }
    assert_eq!(
        cpu.state.pc(),
        CODE,
        "on the exception PC stops at the BKPT"
    );
}

#[test]
fn msr_immediate_sets_mode() {
    let mut cpu = cpu_with(&[
        0xE321_F013, // msr cpsr_c, #0x13  (Supervisor)
        0xE321_F0D3, // msr cpsr_c, #0xD3  (Supervisor, I and F masked)
    ]);
    cpu.state.cpsr.set_mode(rynarmic::Mode::System);

    cpu.step().unwrap();
    assert_eq!(cpu.state.cpsr.mode(), rynarmic::Mode::Supervisor);

    cpu.step().unwrap();
    assert!(cpu.state.cpsr.i() && cpu.state.cpsr.f());
}

#[test]
fn blx_immediate_switches_to_thumb() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xFA00_0002, // blx 0x1010
            0xEAFF_FFFE, // must not execute
        ],
    );
    memory.write_thumb(0x1010, &[0x2005, 0xE7FE]); // movs r0,#5; b .

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);

    cpu.step().unwrap();
    assert!(
        cpu.state.thumb(),
        "BLX should enter Thumb state immediately"
    );
    assert_eq!(cpu.state.pc(), 0x1010);
    assert_eq!(
        cpu.state.reg(14),
        0x1004,
        "the return address is the next ARM instruction"
    );

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 5);
}

#[test]
fn halfword_multiply_treats_halves_as_signed() {
    // As a halfword, 0xFFFF is -1 and not 65535. That distinction only shows on the negative half:
    // the interpreter used to multiply halfwords as unsigned, so SMULbb(-1, 3) gave 196605 instead of -3.
    //
    // The operand mapping, established by probing a reference implementation: SMULxy Rd, Rn, Rm has
    // Rn = bits(3,0) with x = bit5 selecting the halfword, and Rm = bits(11,8) with y = bit6.
    // In the encodings below Rn = r2 and Rm = r1.
    let mut cpu = cpu_with(&[
        0xE160_0182, // smulbb r0, r1, r2: Rn(r2).B × Rm(r1).B
        0xE160_01C2, // smulbt r0, r1, r2: Rn(r2).B × Rm(r1).T
        0xE160_01A2, // smultb r0, r1, r2: Rn(r2).T × Rm(r1).B
        0xE160_01E2, // smultt r0, r1, r2: Rn(r2).T × Rm(r1).T
    ]);
    cpu.state.set_reg(1, 0x8001_FFFF); // Rm: low halfword -1, top halfword -32767
    cpu.state.set_reg(2, 0x8002_0003); // Rn: low halfword 3, top halfword -32766

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), (-3i32) as u32, "SMULbb = 3 × (-1)");

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(0),
        (3i32 * -32767) as u32,
        "SMULbt = Rn.B(3) × Rm.T(-32767)"
    );

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(0),
        ((-32766i32) * -1) as u32,
        "SMULtb = Rn.T(-32766) × Rm.B(-1)"
    );

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(0),
        ((-32766i32) * -32767) as u32,
        "SMULtt = (-32766) × (-32767)"
    );
}

#[test]
fn word_half_multiply_treats_high_half_as_signed() {
    // SMULWy's top halfword has to be read as signed too.
    // The operand mapping, from the same probing: the word is bits(3,0) (Rn),
    // the y halfword is bits(11,8) (Rm) with y = bit6, and the encodings below have Rn = r2, Rm = r1.
    let mut cpu = cpu_with(&[
        0xE120_01A2, // smulwb r0, r1, r2  (Rm's low halfword)
        0xE120_01E2, // smulwt r0, r1, r2  (Rm's top halfword)
    ]);
    cpu.state.set_reg(1, 0x8002_0004); // Rm: low halfword 4, top halfword -32766
    cpu.state.set_reg(2, 0x0001_0000); // Rn: the word 65536

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 4, "SMULWb = (65536 × 4) >> 16");

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(0),
        (((-32766i32) * 65536) >> 16) as u32,
        "SMULWt = (65536 × -32766) >> 16"
    );
}

#[test]
fn smla_sets_q_on_accumulate_overflow() {
    // Per the ARM ARM, SMLAxy and SMLAWy set Q on accumulation overflow; SMULxy has no accumulation and leaves Q alone.
    let mut cpu = cpu_with(&[
        0xE100_3182, // smlabb r0, r1, r2, r3
        0xE160_0182, // smulbb r0, r1, r2
        0xE120_3182, // smlawb r0, r1, r2, r3
    ]);
    cpu.state.set_reg(1, 0x0000_7FFF); // 32767
    cpu.state.set_reg(2, 0x0000_7FFF); // 32767
    cpu.state.set_reg(3, 0x7FFF_FFFF); // push the accumulator up to near INT32_MAX

    cpu.step().unwrap();
    assert!(cpu.state.cpsr.q(), "32767 x 32767 + INT32_MAX has to set Q");
    assert_eq!(cpu.state.reg(0), 0x3FFF_0001u32.wrapping_add(0x7FFF_FFFF));

    // The same product without accumulation must not set Q (cleared first, then tried)
    cpu.state.cpsr.set_q(false);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0x3FFF_0001);
    assert!(
        !cpu.state.cpsr.q(),
        "SMULbb has no accumulation and must not touch Q"
    );

    // SMLAWy follows the same rule (the word in r2 = bits(3,0), the y halfword in r1 = bits(11,8))
    cpu.state.set_reg(1, 0x0000_7FFF);
    cpu.state.set_reg(2, 0x7FFF_FFFF);
    cpu.state.set_reg(3, 0x7FFF_FFFF);
    cpu.state.cpsr.set_q(false);
    cpu.step().unwrap();
    assert!(
        cpu.state.cpsr.q(),
        "SMLAWy sets Q on accumulation overflow as well"
    );
}

#[test]
fn conditional_v5te_instruction_is_skipped() {
    let mut cpu = cpu_with(&[
        0x0000_0192, // muleq r0, r1, r2 -- skipped while Z=0
        0xE1A0_0000, // mov r0, r0
    ]);
    cpu.state.set_reg(1, 6);
    cpu.state.set_reg(2, 7);
    cpu.state.cpsr.set_z(false);

    assert_eq!(cpu.step().unwrap(), StepOutcome::Skipped);
    assert_eq!(cpu.state.reg(0), 0);
}

// ---------------------------------------------------------------------------
// Writing PC: a *load* interworks, a data operation does not (ARMv5TE)
//
// A Thumb function return is `pop {.., pc}`, and the return address's bit 0 is the state bit:
// writing it into PC from a load has to clear bit 0 and set T from it. Those used to be emitted as a
// plain `Jump`, stuffing the address with bit 0 straight into PC and then fetching under the wrong
// state (a host running a real module crashed on an unmapped address because of it).
//
// A **data-processing** write is the opposite and used to be wrong the other way round: `MOV pc, Rm`
// went out as `JumpExchange`, which is the ARMv7 rule. Dynarmic states the split in
// `a32_ir_emitter.cpp` — `ALUWritePC` answers `BXWritePC` only for `ArchVersion() >= 7` in ARM state,
// and `BranchWritePC` for everything else — and this backend is ARMv5TEJ, so `MOV pc, Rm` clears the
// low bits and leaves the instruction set alone. Both halves are pinned below, because a decoder
// change for Thumb-2 will touch exactly this neighbourhood.
// ---------------------------------------------------------------------------

#[test]
fn arm_ldr_pc_interworks() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE591_F000]); // ldr pc, [r1]
    memory.write_words(DATA, &[0x1051]); // odd, so the target state is Thumb
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, DATA);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1050, "bit 0 has to be cleared from PC");
    assert!(cpu.state.thumb(), "bit 0 = 1 means switching to Thumb");
}

#[test]
fn arm_ldm_with_pc_interworks() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE891_8001]); // ldmia r1, {r0, pc}
    memory.write_words(DATA, &[0xDEAD_BEEF, 0x1051]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, DATA);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0xDEAD_BEEF);
    assert_eq!(cpu.state.pc(), 0x1050);
    assert!(cpu.state.thumb());
}

#[test]
fn arm_data_processing_write_to_pc_does_not_interwork() {
    // `mov pc, r1` with an odd value: `BranchWritePC` clears the low bits and leaves T alone, so this
    // stays in ARM state. It used to switch to Thumb (`JumpExchange`), which is ARMv7's rule.
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE1A0_F001]); // mov pc, r1
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, 0x1051);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1050, "the low two bits are cleared");
    assert!(
        !cpu.state.thumb(),
        "a data-processing write to PC does not change the instruction set"
    );

    // Bit 1 as well: ARMv5 calls an unaligned ALU target UNPREDICTABLE, and masking both bits is what
    // `BranchWritePC` documents rather than something to guess at.
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE1A0_F001]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, 0x1052);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1050);
    assert!(!cpu.state.thumb());
}

#[test]
fn thumb_data_processing_write_to_pc_clears_only_bit_0() {
    // The same rule in Thumb state, where only bit 0 is masked: `mov pc, r1` with 0x1053 lands on
    // 0x1052 and stays Thumb. Dynarmic's `BranchWritePC` masks `0xFFFFFFFE` when the T flag is set.
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(CODE, &[0x468F]); // mov pc, r1
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13 | 0x20);
    cpu.state.set_reg(1, 0x1053);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1052);
    assert!(cpu.state.thumb());
}

#[test]
fn thumb_pop_pc_interworks() {
    // pop {pc}: the return value is on top of the stack and bit 0 decides the target state. This is Thumb's function return.
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(CODE, &[0xBD00]); // pop {pc}
    memory.write_words(0x2000, &[0x1051]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13 | 0x20);
    cpu.state.set_reg(13, 0x2000);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1050);
    assert!(cpu.state.thumb());

    // Even, so switch back to ARM (which is what `--apcs=/interwork` relies on for cross-module calls)
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(CODE, &[0xBD00]);
    memory.write_words(0x2000, &[0x1050]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13 | 0x20);
    cpu.state.set_reg(13, 0x2000);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1050);
    assert!(!cpu.state.thumb());
}

/// The same rule through the compiled path, which has its own masking for a dynamic jump target.
///
/// The interpreter and the JIT have to agree here or a run stops being replayable; the mask lives in
/// two different places (`ArmState::branch_target` and `backend/x64.rs`'s `IrOp::Jump`), so this test
/// exists to notice when only one of them is touched.
#[test]
fn arm_data_processing_write_to_pc_does_not_interwork_under_the_jit() {
    let run = |jit: bool| {
        let mut memory = SimpleMemory::new(0, 0x4000);
        memory.write_words(CODE, &[0xE1A0_F001, 0xEAFF_FFFE]); // mov pc, r1 ; b .
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, 0x1051);
        cpu.set_jit_enabled(jit);
        if jit {
            cpu.run_cached(1).unwrap();
            assert!(
                cpu.compiled_block_count() >= 1,
                "the block has to be compiled for this to test the JIT"
            );
        } else {
            cpu.run(1).unwrap();
        }
        (cpu.state.pc(), cpu.state.thumb())
    };
    assert_eq!(run(false), (0x1050, false), "interpreted");
    assert_eq!(run(true), (0x1050, false), "compiled, and it must agree");
}
