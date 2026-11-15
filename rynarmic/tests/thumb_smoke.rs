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
