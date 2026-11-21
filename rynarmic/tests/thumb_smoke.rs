use rynarmic::{Cpu, StepOutcome, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const SPIN: u32 = 0x2000;

fn run_until_pc(cpu: &mut Cpu<SimpleMemory>, target: u32, max: u64) {
    for _ in 0..max {
        if cpu.state.pc() == target {
            return;
        }
        cpu.step().expect("unexpected exception");
    }
    panic!("did not reach {target:#x}, stuck at {:#x}", cpu.state.pc());
}

#[test]
fn thumb_loop_sums() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x2000, // 1000: movs r0, #0
            0x210A, // 1002: movs r1, #10
            0x1808, // 1004: adds r0, r0, r1
            0x3901, // 1006: subs r1, #1
            0xD1FC, // 1008: bne  1004
            0x4770, // 100A: bx   lr
        ],
    );
    memory.write_thumb(SPIN, &[0xE7FE]); // an infinite loop

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(14, SPIN | 1); // the Thumb bit

    run_until_pc(&mut cpu, SPIN, 1000);
    assert_eq!(cpu.state.reg(0), 55);
    assert!(
        cpu.state.thumb(),
        "bx to an odd address should stay in Thumb state"
    );
}

#[test]
fn thumb_push_pop_round_trip() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0xB580, // 1000: push {r7, lr}
            0x2009, // 1002: movs r0, #9
            0xBD80, // 1004: pop  {r7, pc}
        ],
    );
    memory.write_thumb(SPIN, &[0xE7FE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(7, 0x1234);
    cpu.state.set_reg(13, 0x3000);
    cpu.state.set_reg(14, SPIN);

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(13),
        0x3000 - 8,
        "pushing two registers should subtract 8"
    );
    assert_eq!(cpu.memory.peek_u32(0x2FF8), 0x1234);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 9);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(13), 0x3000, "pop should restore SP");
    assert_eq!(cpu.state.reg(7), 0x1234, "pop should restore r7");
    assert_eq!(cpu.state.pc(), SPIN, "pop {{pc}} should jump to LR");
}

#[test]
fn thumb_ldr_literal_reads_aligned_pc() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x4801, // 1000: ldr r0, [pc, #4]  -> 0x1004, aligned, plus 4 = 0x1004
            0xE7FE, // 1002: b .
        ],
    );
    // 0x1000 + 4 aligned is 0x1004, plus 4 is 0x1008
    memory.write_words(0x1008, &[0xDEAD_BEEF]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 0xDEAD_BEEF);
}

#[test]
fn thumb_unconditional_branch() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0xE001, // 1000: b 1006
            0x2001, // 1002: movs r0, #1   (should be skipped)
            0xE7FE, // 1004: b .
            0x2002, // 1006: movs r0, #2
            0xE7FE, // 1008: b .
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1006);
    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 2);
}

#[test]
fn thumb_bl_links_and_branches() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0xF000, 0xF806, // 1000: bl 0x1010
            0xE7FE, // 1004: b . (must not execute)
        ],
    );
    memory.write_thumb(0x1010, &[0x2005, 0xE7FE]); // movs r0, #5; b .

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);

    assert_eq!(
        cpu.step().unwrap(),
        StepOutcome::Skipped,
        "the first halfword is only recorded, never executed"
    );
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1010);
    assert_eq!(
        cpu.state.reg(14),
        0x1005,
        "LR points past the BL and carries the Thumb bit"
    );
    assert!(cpu.state.thumb());

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 5);
}

#[test]
fn thumb_blx_switches_to_arm() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0xF000, 0xE806, // 1000: blx 0x1010
            0xE7FE,
        ],
    );
    memory.write_words(0x1010, &[0xE3A0_0007, 0xEAFF_FFFE]); // mov r0, #7; b .

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);

    cpu.step().unwrap();
    cpu.step().unwrap();
    assert_eq!(cpu.state.pc(), 0x1010);
    assert!(!cpu.state.thumb(), "BLX should switch to ARM state");
    assert_eq!(cpu.state.reg(14), 0x1005);

    cpu.step().unwrap();
    assert_eq!(cpu.state.reg(0), 7);
}

#[test]
fn thumb_blx_register_links_with_the_thumb_bit_set() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x4788, // 1000: blx r1
            0xE7FE, // 1002: b .
        ],
    );
    memory.write_words(0x1010, &[0xE12F_FF1E]); // bx lr

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, 0x1010); // even target: BLX switches to ARM

    cpu.step().unwrap();
    assert_eq!(
        cpu.state.reg(14),
        0x1003,
        "BLX (register) links to the next halfword with the Thumb bit set"
    );
    assert_eq!(cpu.state.pc(), 0x1010);
    assert!(!cpu.state.thumb());

    cpu.step().unwrap(); // bx lr
    assert_eq!(cpu.state.pc(), 0x1002, "the link returns to Thumb code");
    assert!(cpu.state.thumb());
}

#[test]
fn thumb_armv6_extends_widen() {
    // (halfword, input in r1, expected in r0, what)
    let cases: &[(u16, u32, u32, &str)] = &[
        (0xB288, 0x0000_8001, 0x0000_8001, "uxth r0, r1"),
        (0xB2C8, 0x0000_00ff, 0x0000_00ff, "uxtb r0, r1"),
        (0xB208, 0x0000_8001, 0xffff_8001, "sxth r0, r1"),
        (0xB248, 0x0000_0080, 0xffff_ff80, "sxtb r0, r1"),
    ];
    for &(insn, input, expected, what) in cases {
        let mut memory = SimpleMemory::new(0, 0x4000);
        memory.write_thumb(CODE, &[insn, 0xE7FE]);
        let mut cpu = Cpu::new(memory);
        cpu.state.cpsr.set_t(true);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, input);
        cpu.step().unwrap();
        assert_eq!(cpu.state.reg(0), expected, "{what}");
        assert_eq!(cpu.state.pc(), CODE + 2, "{what} must not branch");
    }
}

/// `add r0, pc` reads the address of the instruction plus four — and, unlike `ldr r0,[pc,#x]` and
/// `ADD Rd,PC,#imm`, it does **not** clear bit 1 of that address.
///
/// The difference is not academic. A shipped pack reaches `module_base - 4` (where the loader
/// leaves the pointer to the helper table) with `ldr r0,=K` / `add r0,pc` / `ldr r0,[r0,#0x3c]`,
/// and only the unaligned value wraps there; two bytes on sits a different word. So the fixture
/// puts a distinct word at each of the three places the three candidate readings would land, and
/// the assertion names which one it got.
#[test]
fn thumb_add_pc_reads_the_instruction_address_plus_four_unaligned() {
    const TARGET: u32 = 0x0FFC;
    const ADD_AT: u32 = 0x1002;

    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x4804, // 1000: ldr r0, [pc, #16]  -> the literal at 0x1014
            0x4478, // 1002: add r0, pc
            0x6800, // 1004: ldr r0, [r0]
            0x4770, // 1006: bx  lr
        ],
    );
    // `add` is at 0x1002, so the architected value is 0x1006; pick the literal that wraps to TARGET.
    memory.write_words(0x1014, &[TARGET.wrapping_sub(ADD_AT + 4)]);
    memory.write_words(0x0FF8, &[0xAAAA_0001]); // if PC were the instruction address
    memory.write_words(TARGET, &[0xBBBB_0002]); // if PC is the instruction address plus four

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(14, SPIN | 1);

    run_until_pc(&mut cpu, 0x1004, 100);
    assert_eq!(
        cpu.state.reg(0),
        TARGET,
        "`add r0, pc` should have produced the word-aligned-underneath target, not {:?}",
        cpu.state.reg(0)
    );

    run_until_pc(&mut cpu, SPIN, 100);
    assert_eq!(
        cpu.state.reg(0),
        0xBBBB_0002,
        "the load must land at {TARGET:#x}; 0xAAAA_0001 means PC was the instruction address \
         itself and anything else means it was aligned"
    );
}

/// `add pc, rm` is a branch, and in Thumb it does not switch instruction sets.
///
/// A jump table built as `add r3, pc` / `adds r3, r3, r0` / `ldrh r3, [r3, r0]` / `lsls r3` /
/// `add pc, r3` is how a shipped pack dispatches every case. Decoding the last one as an ordinary
/// `ALU` write to r15 left the block's terminator alone, so the driver ran the instruction *after*
/// the table instead of the case -- on `w52ca_brain-challenge` that skipped a whole initialisation
/// path and faulted 600 000 instructions later at address 0.
#[test]
fn thumb_add_pc_branches_without_exchanging() {
    const CASE: u32 = 0x100E;
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x2308, // 1000: movs r3, #8
            0x449F, // 1002: add  pc, r3   -> (0x1002 + 4) + 8
            0x2001, // 1004: movs r0, #1   (the fall-through: must not run)
            0x4770, // 1006: bx   lr
        ],
    );
    memory.write_thumb(CASE, &[0x2002, 0x4770]); // 100E: movs r0, #2 ; bx lr
    memory.write_thumb(SPIN, &[0xE7FE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(14, SPIN | 1);

    run_until_pc(&mut cpu, SPIN, 100);
    assert_eq!(
        cpu.state.reg(0),
        2,
        "`add pc, r3` must branch to {CASE:#x}; 1 means it fell through the table"
    );
    assert!(cpu.state.thumb(), "`add pc, rm` does not exchange");
}

/// The same rule for `mov pc, rm`: a branch, not a write to a general register. Unlike `add pc, rm`
/// there is no PC bias here -- the target *is* the register's value.
#[test]
fn thumb_mov_pc_branches_without_exchanging() {
    const CASE: u32 = 0x40;
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x2140, // 1000: movs r1, #0x40
            0x468F, // 1002: mov  pc, r1   -> 0x40
            0x2001, // 1004: movs r0, #1   (must not run)
            0x4770, // 1006: bx   lr
        ],
    );
    memory.write_thumb(CASE, &[0x2002, 0x4770]); // 0040: movs r0, #2 ; bx lr
    memory.write_thumb(SPIN, &[0xE7FE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(14, SPIN | 1);

    run_until_pc(&mut cpu, SPIN, 100);
    assert_eq!(
        cpu.state.reg(0),
        2,
        "`mov pc, r1` must branch to {CASE:#x}; 1 means it fell through"
    );
    assert!(cpu.state.thumb(), "`mov pc, rm` does not exchange");
}

/// `1011 0z01` is `CBZ`/`CBNZ`, and `ADD/SUB SP` is only `1011 0000`.
///
/// The `misc` group answered both `0b0000` and `0b0001` with the SP adjustment, so a `CBZ` — an
/// ARMv6T2 instruction this decoder does not implement — used to move the stack pointer by whatever
/// `bits[6:0]` happened to say. Nothing faults, the stack just slides, and the fault turns up
/// hundreds of instructions later somewhere unrelated. Refusing is the only honest answer.
#[test]
fn cbz_is_refused_rather_than_answered_as_a_stack_adjustment() {
    use rynarmic::DecodeError;
    for raw in 0xB100u16..=0xB1FF {
        match rynarmic::decoder::thumb::decode(raw, CODE, true) {
            Err(DecodeError::Unsupported { reason, .. }) => {
                assert!(reason.contains("CBZ"), "{raw:#06x}: {reason}");
            }
            other => panic!("{raw:#06x} must not decode: {other:?}"),
        }
    }
    // The two SP forms still do, and they move SP by `imm7 << 2`.
    for raw in [0xB000u16, 0xB080, 0xB0FF] {
        assert!(
            rynarmic::decoder::thumb::decode(raw, CODE, true).is_ok(),
            "{raw:#06x} is ADD/SUB SP and must still decode"
        );
    }
}
