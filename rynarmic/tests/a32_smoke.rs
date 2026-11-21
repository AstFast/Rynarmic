use rynarmic::{Cpsr, Cpu, StepOutcome, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const SPIN: u32 = 0x2000;
const DATA: u32 = 0x3000;

/// Iteratively sums 10+9+...+1 = 55 and writes the result back to memory.
fn sum_program() -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0x0000, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE3A0_0000, // 1000: mov r0, #0
            0xE3A0_100A, // 1004: mov r1, #10
            0xE080_0001, // 1008: add r0, r0, r1
            0xE251_1001, // 100C: subs r1, r1, #1
            0xE351_0000, // 1010: cmp  r1, #0
            0x1AFF_FFFB, // 1014: bne  1008
            0xE582_0000, // 1018: str  r0, [r2]
            0xE12F_FF1E, // 101C: bx   lr
        ],
    );
    // The return point: an infinite loop, so the test can observe a settled state
    memory.write_words(SPIN, &[0xEAFF_FFFE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(2, DATA);
    cpu.state.set_reg(14, SPIN);
    cpu
}

fn run_until_pc(cpu: &mut Cpu<SimpleMemory>, target: u32, max: u64) {
    for _ in 0..max {
        if cpu.state.pc() == target {
            return;
        }
        cpu.step().expect("unexpected exception");
    }
    panic!("did not reach {target:#x}, stuck at {:#x}", cpu.state.pc());
}

/// Executes one A32 instruction and returns the CPU afterwards.
fn run_one(inst: u32, r1: u32, r2: u32, carry: bool) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[inst, 0xEAFF_FFFE]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, r1);
    cpu.state.set_reg(2, r2);
    let bits = (cpu.state.cpsr.bits() & !Cpsr::C) | if carry { Cpsr::C } else { 0 };
    cpu.state.cpsr = Cpsr::from_bits(bits);
    cpu.step().expect("unexpected exception");
    cpu
}

#[test]
fn a32_loop_sums_and_stores() {
    let mut cpu = sum_program();
    run_until_pc(&mut cpu, SPIN, 1000);

    assert_eq!(cpu.state.reg(0), 55);
    assert_eq!(cpu.memory.peek_u32(DATA), 55);
    assert!(cpu.state.cpsr.z(), "the final cmp r1, #0 should set Z");
}

#[test]
fn a32_cmp_sets_z_when_equal() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE3A0_0005, // mov r0, #5
            0xE350_0005, // cmp r0, #5
            0xEAFF_FFFE, // b .
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.step().unwrap();
    cpu.step().unwrap();
    assert!(cpu.state.cpsr.z());
    assert!(!cpu.state.cpsr.n());
}

#[test]
fn a32_add_sets_carry_and_overflow() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE3E0_0000, // mvn r0, #0        ; r0 = 0xFFFF_FFFF
            0xE290_0001, // adds r0, r0, #1   ; wraps around, so C=1 and Z=1
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.step().unwrap();
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0);
    assert!(cpu.state.cpsr.c());
    assert!(cpu.state.cpsr.z());
    assert!(
        !cpu.state.cpsr.v(),
        "0xFFFFFFFF + 1 is not a signed overflow"
    );
}

#[test]
fn a32_conditional_instruction_is_skipped() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0x03A0_0007, // moveq r0, #7  ; skipped while Z=0
            0xE3A0_1001, // mov r1, #1
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    assert_eq!(cpu.step().unwrap(), StepOutcome::Skipped);
    assert_eq!(cpu.state.reg(0), 0);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(1), 1);
}

#[test]
fn a32_ldm_stm_round_trip() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE8A9_000E, // stmia r9!, {r1, r2, r3}
            0xE1A0_0009, // mov   r0, r9
            0xE240_000C, // sub   r0, r0, #12
            0xE890_0070, // ldmia r0, {r4, r5, r6}
            0xEAFF_FFFE,
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(9, DATA);
    cpu.state.set_reg(1, 0x11);
    cpu.state.set_reg(2, 0x22);
    cpu.state.set_reg(3, 0x33);

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(9),
        DATA + 12,
        "STM writeback should advance by 12 bytes"
    );
    assert_eq!(cpu.memory.peek_u32(DATA + 8), 0x33);

    cpu.step().unwrap(); // mov r0, r9
    cpu.step().unwrap(); // sub r0, r0, #12
    cpu.step().unwrap(); // ldmia
    assert_eq!(
        (cpu.state.reg(4), cpu.state.reg(5), cpu.state.reg(6)),
        (0x11, 0x22, 0x33)
    );
}

#[test]
fn a32_unsupported_instruction_reports_exception() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    // vmov s0, r0: a VFP instruction outside the covered set, which must be reported rather than silently miscalculated
    memory.write_words(CODE, &[0xEE00_0A10]);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    let result = cpu.step();
    assert!(
        result.is_err(),
        "a VFP instruction should raise, got {result:?}"
    );
    assert_eq!(
        cpu.state.pc(),
        CODE,
        "on an exception PC stops at the failing instruction"
    );
}

#[test]
fn a32_register_shift_lsr_asr_ror() {
    // movs r0, r1, lsr r2
    let cpu = run_one(0xE1B0_0231, 0x8000_0001, 1, false);
    assert_eq!(cpu.state.reg(0), 0x4000_0000);
    assert!(
        cpu.state.cpsr.c(),
        "LSR's carry is the bit shifted out, bit 0"
    );

    // movs r0, r1, asr r2
    let cpu = run_one(0xE1B0_0251, 0x8000_0000, 4, false);
    assert_eq!(cpu.state.reg(0), 0xF800_0000);
    assert!(
        !cpu.state.cpsr.c(),
        "ASR's carry is the last bit shifted out, bit 3"
    );

    // movs r0, r1, ror r2
    let cpu = run_one(0xE1B0_0271, 0x0000_0001, 1, false);
    assert_eq!(cpu.state.reg(0), 0x8000_0000);
    assert!(cpu.state.cpsr.c(), "ROR's carry is the result's bit 31");

    // A register shift amount of 0 means "no shift", not RRX
    let cpu = run_one(0xE1B0_0271, 0x0000_0001, 0, false);
    assert_eq!(cpu.state.reg(0), 1);
    assert!(!cpu.state.cpsr.c(), "a register shift of 0 leaves C alone");
}

#[test]
fn a32_immediate_shift_zero_semantics() {
    // movs r0, r1, lsr #0 == lsr #32
    let cpu = run_one(0xE1B0_0021, 0x8000_0000, 0, false);
    assert_eq!(cpu.state.reg(0), 0);
    assert!(cpu.state.cpsr.c());
    assert!(cpu.state.cpsr.z());

    // movs r0, r1, asr #0 == asr #32
    let cpu = run_one(0xE1B0_0041, 0x8000_0000, 0, false);
    assert_eq!(cpu.state.reg(0), 0xFFFF_FFFF);
    assert!(cpu.state.cpsr.c());

    // movs r0, r1, ror #0 is rrx: C enters the top bit, once as 1 and once as 0
    let cpu = run_one(0xE1B0_0061, 0x0000_0001, 0, true);
    assert_eq!(cpu.state.reg(0), 0x8000_0000);
    assert!(cpu.state.cpsr.c());

    let cpu = run_one(0xE1B0_0061, 0x0000_0001, 0, false);
    assert_eq!(cpu.state.reg(0), 0);
    assert!(
        cpu.state.cpsr.c(),
        "RRX's carry out is the old value's bit 0"
    );

    // movs r0, r1, lsl #0: no shift, and C keeps its value
    let cpu = run_one(0xE1B0_0001, 0x8000_0000, 0, false);
    assert_eq!(cpu.state.reg(0), 0x8000_0000);
    assert!(!cpu.state.cpsr.c(), "LSL #0 does not write C");
}

#[test]
fn a32_register_offset_shift_is_applied() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, &[0xE791_0122]); // ldr r0, [r1, r2, lsr #2]
    memory.write_words(
        DATA,
        &[
            0xAAAA_0001,
            0xBBBB_0002,
            0xCCCC_0003,
            0xDDDD_0004,
            0xEEEE_0005,
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, DATA);
    cpu.state.set_reg(2, 16);
    cpu.step().unwrap();
    // 16 >> 2 == 4; using LSL by mistake would read DATA + 64
    assert_eq!(cpu.state.reg(0), 0xBBBB_0002);
}

/// `sxth` / `sxtb` / `uxth` / `uxtb` — ARMv6's sign/zero extend, which a shipped pack executes on
/// its module-init path (`sh003_gachinko`, `so004_gachinko`). The encoding lives in the load/store
/// group's bits[27:25] but in none of its fields, so reading it as a transfer both refused the pack
/// under this decoder and silently ignored it under the host's interpreter.
///
/// One word per case, each `mov`-ing the input in and checking the extended result.
#[test]
fn armv6_extend_instructions_widen_correctly() {
    // mvn r0, #0 gives 0xFFFF_FFFF; `lsr r0, #0` style tricks are unnecessary here because the
    // inputs are small enough to load with `mov` and then shifted into place by the instruction
    // under test.
    let cases: &[(u32, u32, u32, &str)] = &[
        // (instruction, input, expected, what)
        (
            0xE6BF_1070,
            0x0000_8001,
            0xFFFF_8001,
            "sxth 0x8001 -> 0xFFFF8001",
        ),
        (
            0xE6BF_1070,
            0x0000_7FFF,
            0x0000_7FFF,
            "sxth 0x7FFF stays positive",
        ),
        (0xE6AF_1070, 0x0000_0080, 0xFFFF_FF80, "sxtb 0x80 -> -128"),
        (0xE6FF_1070, 0x0000_8001, 0x0000_8001, "uxth zero-extends"),
        (0xE6EF_1070, 0x0000_00FF, 0x0000_00FF, "uxtb zero-extends"),
    ];

    for &(insn, input, expected, what) in cases {
        let mut memory = SimpleMemory::new(0x0000, 0x4000);
        // The input is loaded by `mov r0, #imm8` when it fits and by `ldr r0, [pc, #4]` otherwise;
        // both are one word wide, so the instruction under test is always at CODE + 4.
        if input <= 0xFF {
            memory.write_words(CODE, &[0xE3A0_0000 | input, insn, 0xE12F_FF1E]);
        } else {
            memory.write_words(CODE, &[0xE59F_0004, insn, 0xE12F_FF1E, input]);
        }
        memory.write_words(SPIN, &[0xEAFF_FFFE]);

        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(14, SPIN);
        run_until_pc(&mut cpu, SPIN, 100);
        assert_eq!(
            cpu.state.reg(1),
            expected,
            "{what}: r1 = {:#010x}",
            cpu.state.reg(1)
        );
    }
}
