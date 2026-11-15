//! Correctness tests for basic-block translation and caching.
//!
//! The core method is **differencing**: run one program through the single-step path (`run`) and the block cache (`run_cached`) and
//! require identical registers, flags and memory. "Conditions decided at run time" is checked separately, because that is where
//! the block cache goes wrong most easily: bake the condition into the block and every later run takes the wrong path.

use rynarmic::{Cpsr, Cpu, Mode, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;

fn arm_cpu(words: &[u32], len: usize) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, len);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu
}

fn thumb_cpu(halfwords: &[u16]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(CODE, halfwords);
    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu
}

/// Returns (registers, cpsr, memory snapshot)
fn snapshot(cpu: &Cpu<SimpleMemory>, words: usize) -> ([u32; 16], u32, Vec<u32>) {
    let mut mem = Vec::with_capacity(words);
    for i in 0..words {
        mem.push(cpu.memory.peek_u32(DATA + (i as u32) * 4));
    }
    (cpu.state.r, cpu.state.cpsr.bits(), mem)
}

#[test]
fn arm_block_execution_matches_step_execution() {
    // 10+9+...+1 = 55, written back to memory
    let program = [
        0xE3A0_0000, // mov r0, #0
        0xE3A0_100A, // mov r1, #10
        0xE080_0001, // add r0, r0, r1
        0xE251_1001, // subs r1, r1, #1
        0xE351_0000, // cmp r1, #0
        0x1AFF_FFFB, // bne 0x1008
        0xE582_0000, // str r0, [r2]
        0xEAFF_FFFE, // b .
    ];

    let mut stepped = arm_cpu(&program, 0x4000);
    stepped.state.set_reg(2, DATA);
    stepped.run(200).unwrap();

    let mut cached = arm_cpu(&program, 0x4000);
    cached.state.set_reg(2, DATA);
    cached.run_cached(200).unwrap();

    assert_eq!(snapshot(&stepped, 4), snapshot(&cached, 4));
    assert_eq!(cached.state.reg(0), 55);
    assert!(
        cached.cached_block_count() > 0,
        "basic blocks should have been cached"
    );
}

#[test]
fn thumb_block_execution_matches_step_execution() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x2000, // movs r0, #0
            0x210A, // movs r1, #10
            0x1808, // adds r0, r0, r1
            0x3901, // subs r1, #1
            0xD1FC, // bne 0x1004
            0xE7FE, // b .
        ],
    );

    let mut stepped = Cpu::new(memory.clone());
    stepped.state.cpsr.set_t(true);
    stepped.state.set_pc(CODE);
    stepped.run(100).unwrap();

    let mut cached = Cpu::new(memory);
    cached.state.cpsr.set_t(true);
    cached.state.set_pc(CODE);
    cached.run_cached(100).unwrap();

    assert_eq!(stepped.state.r, cached.state.r);
    assert_eq!(stepped.state.cpsr.bits(), cached.state.cpsr.bits());
    assert_eq!(cached.state.reg(0), 55);
}

#[test]
fn conditional_branch_is_evaluated_at_runtime() {
    let program = [
        0xE3B0_0000, // movs r0, #0   -> Z=1
        0x1A00_0001, // bne 0x1010    -> not taken
        0xE3A0_0001, // mov r0, #1
        0xEA00_0002, // b 0x1018
        0xE3A0_0002, // 0x1010: mov r0, #2
        0xEA00_0000, // b 0x1018
        0xEAFF_FFFE, // 0x1018: b .
    ];

    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.run_cached(8).unwrap();
    assert_eq!(cpu.state.reg(0), 1, "BNE must not jump while Z=1");
    let blocks = cpu.cached_block_count();
    assert!(blocks >= 2);

    // Back at the same (cached) conditional branch, but now Z=0: it must decide afresh and jump
    cpu.state.set_pc(0x1004);
    cpu.state.cpsr.set_z(false);
    cpu.run_cached(2).unwrap();
    assert_eq!(cpu.state.reg(0), 2, "the same block must jump while Z=0");
    // The targets add blocks, but the conditional branch block itself must be reused (at most two new blocks)
    assert!(
        cpu.cached_block_count() <= blocks + 2,
        "unexpected block growth: {} -> {}",
        blocks,
        cpu.cached_block_count()
    );
}

#[test]
fn conditional_data_instruction_is_not_cached_as_taken() {
    let program = [
        0x03A0_0007, // moveq r0, #7
        0xEAFF_FFFE, // b .
    ];

    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.state.cpsr.set_z(false);
    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.reg(0), 0, "MOVEQ should be skipped while Z=0");

    // The same address and the same cache entry: once the condition holds it must execute
    cpu.state.set_pc(CODE);
    cpu.state.cpsr.set_z(true);
    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.reg(0), 7, "MOVEQ must execute while Z=1");
}

#[test]
fn conditional_instructions_do_not_fragment_blocks() {
    // A conditionally executed instruction now expands to a guard plus operations and can stay in the block.
    // They used to shatter blocks down to one instruction, and such instructions are a large share of A32 code.
    let block = translate_block(&[
        0xE350_0000, // cmp   r0, #0
        0x03A0_1001, // moveq r1, #1
        0x1282_2001, // addne r2, r2, #1
        0xEAFF_FFFE, // b .
    ]);
    assert_eq!(
        block.instruction_count, 4,
        "the whole sequence should form one block"
    );
    assert!(
        block.has_cond,
        "the block should be marked as containing a guard"
    );

    // The guard's shape: cond and skip come in a pair, and skip covers every operation the instruction expanded to
    let guards: Vec<(u32, u8)> = block
        .ops
        .iter()
        .filter_map(|op| match op {
            rynarmic::IrOp::Cond { cond, skip } => Some((*cond, *skip)),
            _ => None,
        })
        .collect();
    assert_eq!(
        guards,
        vec![(0x0, 1), (0x1, 1)],
        "one guard for moveq and one for addne"
    );
}

#[test]
fn conditional_store_is_skipped_when_condition_fails() {
    // Note that `cmp r0, #0` overwrites the flags, so r0 decides the condition: r0 != 0 gives Z=0, r0 = 0 gives Z=1.
    let program = [
        0xE350_0000, // cmp   r0, #0
        0x0581_2000, // streq r2, [r1]
        0xEAFF_FFFE, // b .
    ];

    let mut skipped = arm_cpu(&program, 0x4000);
    skipped.state.set_reg(1, DATA);
    skipped.state.set_reg(0, 1);
    skipped.state.set_reg(2, 0xAAAA_5555);
    skipped.run_cached(2).unwrap();
    assert_eq!(
        skipped.memory.peek_u32(DATA),
        0,
        "STREQ must not write memory while Z=0"
    );

    let mut taken = arm_cpu(&program, 0x4000);
    taken.state.set_reg(1, DATA);
    taken.state.set_reg(0, 0);
    taken.state.set_reg(2, 0xAAAA_5555);
    taken.run_cached(2).unwrap();
    assert_eq!(
        taken.memory.peek_u32(DATA),
        0xAAAA_5555,
        "it must write while Z=1"
    );
}

#[test]
fn conditional_write_to_pc_acts_as_conditional_branch() {
    // moveq pc, lr — a conditionally executed branch: it jumps when the condition holds and falls through when it does not.
    let program = [
        0xE350_0000, // cmp    r0, #0
        0x01A0_F00E, // moveq  pc, lr
        0xE3A0_5001, // mov    r5, #1
        0xEAFF_FFFE, // b .
    ];

    // `cmp r0, #0` decides: r0 = 0 gives Z=1 (jump), r0 != 0 gives Z=0 (fall through)
    let mut taken = arm_cpu(&program, 0x4000);
    taken.state.set_reg(14, 0x1020);
    taken.state.set_reg(0, 0);
    taken.run_cached(2).unwrap();
    assert_eq!(taken.state.pc(), 0x1020, "MOV PC should jump while Z=1");
    assert_eq!(taken.state.reg(5), 0);

    let mut not_taken = arm_cpu(&program, 0x4000);
    not_taken.state.set_reg(14, 0x1020);
    not_taken.state.set_reg(0, 1);
    not_taken.run_cached(2).unwrap();
    assert_eq!(
        not_taken.state.pc(),
        0x1008,
        "while Z=0 it should continue to the next instruction"
    );
}

#[test]
fn conditional_bl_does_not_clobber_lr_when_condition_fails() {
    // `BL<cond>` has two observable effects, writing LR and jumping, and both must share one condition;
    // otherwise a failed condition silently rewrites LR — which is what happens when a compiler puts BL at the end of a conditional block.
    let program = [
        0xE350_0000, // cmp  r0, #0
        0x0B00_0005, // bleq 0x1020
        0xEAFF_FFFE, // b .
    ];

    let mut skipped = arm_cpu(&program, 0x4000);
    skipped.state.set_reg(0, 1); // Z=0
    skipped.state.set_reg(14, 0xDEAD_BEEF);
    skipped.run_cached(2).unwrap();
    assert_eq!(
        skipped.state.reg(14),
        0xDEAD_BEEF,
        "a failed BLEQ must not write LR"
    );
    assert_eq!(skipped.state.pc(), 0x1008);

    let mut taken = arm_cpu(&program, 0x4000);
    taken.state.set_reg(0, 0); // Z=1
    taken.state.set_reg(14, 0xDEAD_BEEF);
    taken.run_cached(2).unwrap();
    assert_eq!(
        taken.state.reg(14),
        0x1008,
        "with the condition holding, LR is the address after BL"
    );
    assert_eq!(taken.state.pc(), 0x1020);
}

#[test]
fn skipped_conditional_instruction_has_no_side_effects() {
    // A skipped conditional instruction may write neither registers nor flags, and that includes conditional test instructions.
    let program = [
        0xE3A0_0000, // mov   r0, #0
        0xE350_0000, // cmp   r0, #0   -> Z=1
        0x03A0_1005, // moveq r1, #5 (executes)
        0x13A0_2005, // movne r2, #5 (skipped)
        0x1350_0001, // cmpne r0, #1 (skipped: Z must stay 1)
        0x03A0_3009, // moveq r3, #9 (executes)
        0xEAFF_FFFE, // b .
    ];

    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.run_cached(6).unwrap();
    assert_eq!(cpu.state.reg(1), 5);
    assert_eq!(cpu.state.reg(2), 0, "MOVNE should be skipped");
    assert!(
        cpu.state.cpsr.z(),
        "a skipped CMPNE must not change the flags"
    );
    assert_eq!(cpu.state.reg(3), 9, "Z is still 1, so MOVEQ should execute");
}

#[test]
fn thumb_bl_pair_inside_block_links_correctly() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0xF000, 0xF806, // bl 0x1010
            0xE7FE,
        ],
    );
    memory.write_thumb(0x1010, &[0x2005, 0xE7FE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);

    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.pc(), 0x1010);
    assert_eq!(cpu.state.reg(14), 0x1005);
}

#[test]
fn code_cache_must_be_invalidated_after_self_modification() {
    let program = [
        0xE3A0_0001, // mov r0, #1
        0xEAFF_FFFE, // b .
    ];
    let mut cpu = arm_cpu(&program, 0x4000);

    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.reg(0), 1);

    // Self-modifying code: replace the first instruction with mov r0, #2
    cpu.memory.write_words(CODE, &[0xE3A0_0002]);
    cpu.state.set_pc(CODE);
    cpu.invalidate_code_cache();
    assert_eq!(cpu.cached_block_count(), 0);

    cpu.run_cached(1).unwrap();
    assert_eq!(
        cpu.state.reg(0),
        2,
        "the new instruction must run once the cache is invalidated"
    );
}

#[test]
fn undefined_instruction_mid_block_keeps_earlier_instructions() {
    // The first instruction in the block is legal and the second is not: what comes before must still execute,
    // and the exception address must be the illegal instruction itself, not the block start.
    let program = [
        0xE3A0_0007, // mov r0, #7
        0xEE00_0A10, // vmov s0, r0 (VFP, not implemented)
    ];

    let mut stepped = arm_cpu(&program, 0x4000);
    let step_result = stepped.run(2);
    assert!(step_result.is_err());
    assert_eq!(
        stepped.state.reg(0),
        7,
        "the single-step path should execute the mov first"
    );
    assert_eq!(stepped.state.pc(), 0x1004);

    let mut cached = arm_cpu(&program, 0x4000);
    let cached_result = cached.run_cached(2);
    assert!(
        cached_result.is_err(),
        "the block path should report the exception too"
    );
    assert_eq!(
        cached.state.reg(0),
        7,
        "the block path must execute the mov first as well"
    );
    assert_eq!(
        cached.state.pc(),
        0x1004,
        "the exception should point at the illegal instruction"
    );
}

#[test]
fn exceptions_report_the_failing_instruction_address() {
    let program = [
        0xE3A0_0001, // mov r0, #1
        0xE120_0070, // bkpt #0
    ];
    let mut cpu = arm_cpu(&program, 0x4000);
    let result = cpu.run_cached(4);
    assert!(result.is_err(), "BKPT should come back as an exception");
    assert_eq!(
        cpu.state.pc(),
        0x1004,
        "on the exception PC stops at the BKPT"
    );
}

#[test]
fn block_translation_stops_at_instruction_limit() {
    // 32 movs followed by a `b .`: the block should be cut off at 32
    let mut words = vec![0xE1A0_0000u32; 40];
    words[39] = 0xEAFF_FFFE;
    let mut cpu = arm_cpu(&words, 0x4000);

    let block = match rynarmic::block::translate(&mut cpu.memory, CODE, false).unwrap() {
        rynarmic::block::Translation::Block(block) => block,
        other => panic!("expected a basic block, got {other:?}"),
    };
    assert_eq!(block.instruction_count, 32);
    assert_eq!(block.next_pc, CODE + 32 * 4);
}

#[test]
fn pc_relative_loads_are_folded_into_blocks() {
    // A Thumb constant literal load is PC-relative. Once the PC is folded into an absolute address at translation time,
    // such instructions can safely enter a block — the key change that keeps real Thumb code off the single-step path.
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x4801, // 1000: ldr r0, [pc, #4]  -> (0x1004) + 4 = 0x1008
            0xE7FE, // 1002: b .
        ],
    );
    memory.write_words(0x1008, &[0x5A5A_A5A5]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);

    cpu.run_cached(1).unwrap();
    assert_eq!(
        cpu.state.reg(0),
        0x5A5A_A5A5,
        "a PC-relative load must read the right address"
    );
    assert!(
        cpu.cached_block_count() >= 1,
        "folding should let it form a block"
    );
}

#[test]
fn arm_pc_operand_is_folded_into_immediate() {
    // add r0, pc, #8: r15 inside a block is stale, so it has to be folded to (pc+8)+8 at translation time
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE28F_0008, // add r0, pc, #8  -> 0x1000 + 8 + 8 = 0x1010
            0xEAFF_FFFE, // b .
        ],
    );

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.reg(0), 0x1010);
    assert!(cpu.cached_block_count() >= 1);
}

fn translate_block(words: &[u32]) -> rynarmic::BasicBlock {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    match rynarmic::block::translate(&mut memory, CODE, false).unwrap() {
        rynarmic::block::Translation::Block(block) => block,
        other => panic!("expected a basic block, got {other:?}"),
    }
}

#[test]
fn dead_flag_write_is_eliminated() {
    // The flags from adds r0, r0, #1 are overwritten by subs with no read in between, so they may be dropped
    let block = translate_block(&[
        0xE290_0001, // adds r0, r0, #1
        0xE251_1001, // subs r1, r1, #1
        0x1AFF_FFFD, // bne 0x1000
    ]);
    match block.ops[0] {
        rynarmic::IrOp::Alu {
            set_flags: false, ..
        } => {}
        other => panic!("the adds flags should have been eliminated, got {other:?}"),
    }
}

#[test]
fn preserved_carry_keeps_earlier_flag_write_alive() {
    // movs r2, r3 has a shift amount of 0, so it **does not write C**, which leaves the C from the earlier adds live.
    // (This is where dead-flag analysis goes wrong most easily: treating every logical instruction as writing C deletes the earlier carry.)
    let block = translate_block(&[
        0xE290_0001, // adds r0, r0, #1
        0xE1B0_2003, // movs r2, r3 (shift 0, C keeps its value)
        0x2A00_0000, // bcs 0x1010 (reads C)
    ]);

    // The invariant that matters: adds's carry write has to survive. If the analysis takes movs for "writes C",
    // adds is judged dead and the BCS afterwards reads a stale carry.
    match block.ops[0] {
        rynarmic::IrOp::Alu {
            op: rynarmic::AluOp::Add,
            set_flags: true,
            ..
        } => {}
        other => panic!("adds's carry is still used and must not be eliminated: {other:?}"),
    }
    // movs still has to write N and Z (the flags at a block exit are live for a successor, so they are kept conservatively)
    match block.ops[1] {
        rynarmic::IrOp::Alu {
            op: rynarmic::AluOp::Mov,
            set_flags: true,
            ..
        } => {}
        other => panic!("movs's N and Z writes are still live at the block exit: {other:?}"),
    }
}

#[test]
fn dead_logical_flag_write_is_eliminated() {
    // movs's N and Z are overwritten by subs with no read in between, so they may be dropped
    let block = translate_block(&[
        0xE1B0_2003, // movs r2, r3
        0xE254_4001, // subs r4, r4, #1
        0x1AFF_FFFC, // bne (reads Z)
    ]);
    match block.ops[0] {
        rynarmic::IrOp::Alu {
            op: rynarmic::AluOp::Mov,
            set_flags: false,
            ..
        } => {}
        other => panic!("movs's flag write is overwritten and should be eliminated: {other:?}"),
    }
}

#[test]
fn mode_and_flags_survive_block_execution() {
    let program = [
        0xE321_F013, // msr cpsr_c, #0x13
        0xEAFF_FFFE, // b .
    ];
    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.state.cpsr.set_mode(Mode::System);
    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.cpsr.mode(), Mode::Supervisor);
}

#[test]
fn bx_pc_uses_instruction_address_in_blocks() {
    // bx pc reads the current instruction address plus 8; inside a block r15 is frozen at the block start, so it has to single-step
    let program = [
        0xE1A0_0000, // mov r0, r0
        0xE12F_FF1F, // bx pc
    ];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&program, 0x4000);
        stepped.run(2).unwrap();

        let mut cached = arm_cpu(&program, 0x4000);
        cached.set_jit_enabled(jit);
        cached.run_cached(2).unwrap();

        assert_eq!(stepped.state.pc(), CODE + 0xC);
        assert_eq!(cached.state.pc(), CODE + 0xC, "jit={jit}");
        assert_eq!(snapshot(&stepped, 0), snapshot(&cached, 0), "jit={jit}");
    }
}

#[test]
fn thumb_bx_pc_uses_instruction_address_in_blocks() {
    for jit in [true, false] {
        let mut stepped = thumb_cpu(&[0x4778, 0xE7FE]); // bx pc ; b .
        stepped.run(1).unwrap();

        let mut cached = thumb_cpu(&[0x4778, 0xE7FE]);
        cached.set_jit_enabled(jit);
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.state.pc(), CODE + 4);
        assert_eq!(cached.state.pc(), CODE + 4, "jit={jit}");
        assert_eq!(snapshot(&stepped, 0), snapshot(&cached, 0), "jit={jit}");
    }
}

#[test]
fn msr_control_bit_switch_ends_block() {
    // After an MSR changes the T bit the next instruction must be fetched as Thumb, so the block cannot keep decoding as ARM
    for jit in [true, false] {
        let mut stepped_memory = SimpleMemory::new(0, 0x4000);
        stepped_memory.write_words(CODE, &[0xE321_F030]); // msr cpsr_c, #0x30 (T=1)
        stepped_memory.write_thumb(CODE + 4, &[0x2001, 0xE7FE]); // movs r0, #1 ; b .
        let mut stepped = Cpu::new(stepped_memory);
        stepped.state.set_pc(CODE);
        stepped.run(2).unwrap();

        let mut cached_memory = SimpleMemory::new(0, 0x4000);
        cached_memory.write_words(CODE, &[0xE321_F030]);
        cached_memory.write_thumb(CODE + 4, &[0x2001, 0xE7FE]);
        let mut cached = Cpu::new(cached_memory);
        cached.set_jit_enabled(jit);
        cached.state.set_pc(CODE);
        cached.run_cached(2).unwrap();

        assert_eq!(cached.state.reg(0), 1, "jit={jit}");
        assert!(cached.state.thumb(), "jit={jit}");
        assert_eq!(cached.state.pc(), CODE + 6, "jit={jit}");
        assert_eq!(snapshot(&stepped, 0), snapshot(&cached, 0), "jit={jit}");
    }
}

#[test]
fn register_shift_carry_write_is_not_eliminated() {
    // muls overwrites movs's N and Z, but strcs still reads C, so movs's flag write cannot be dropped whole
    let program = [
        0xE099_800A, // adds r8, r9, r10 (sets C=1 first)
        0xE1B0_0251, // movs r0, r1, asr r2
        0xE015_0796, // muls r5, r6, r7 (writes N and Z only)
        0x2584_3000, // strcs r3, [r4] (reads C)
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        // r2 != 0: movs writes a new C; r2 == 0: movs leaves the earlier C as it was
        for (r1, r2) in [(1u32, 1u32), (0x8000_0000, 0)] {
            let mut stepped = arm_cpu(&program, 0x4000);
            let mut cached = arm_cpu(&program, 0x4000);
            for cpu in [&mut stepped, &mut cached] {
                cpu.state.set_reg(1, r1);
                cpu.state.set_reg(2, r2);
                cpu.state.set_reg(3, 0x1234);
                cpu.state.set_reg(4, DATA);
                cpu.state.set_reg(6, 0x8000_0000);
                cpu.state.set_reg(7, 1);
                cpu.state.set_reg(9, 0xFFFF_FFFF);
                cpu.state.set_reg(10, 1);
                cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::C);
            }
            cached.set_jit_enabled(jit);

            stepped.run(4).unwrap();
            cached.run_cached(4).unwrap();

            assert_eq!(stepped.memory.peek_u32(DATA), 0x1234, "jit={jit} r2={r2}");
            assert_eq!(cached.memory.peek_u32(DATA), 0x1234, "jit={jit} r2={r2}");
            assert_eq!(
                snapshot(&stepped, 1),
                snapshot(&cached, 1),
                "jit={jit} r2={r2}"
            );
        }
    }
}

#[test]
fn rrx_carry_read_keeps_earlier_flag_write_alive() {
    // RRX depends on the C the first adds produced, so that write must survive until a later adds overwrites C
    let program = [
        0xE096_5007, // adds r5, r6, r7 (C=1)
        0xE1A0_0061, // mov r0, r1, ror #0 (RRX, reads C)
        0xE099_800A, // adds r8, r9, r10 (C=0)
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&program, 0x4000);
        let mut cached = arm_cpu(&program, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(1, 0);
            cpu.state.set_reg(6, 0xFFFF_FFFF);
            cpu.state.set_reg(7, 1);
            cpu.state.set_reg(9, 0);
            cpu.state.set_reg(10, 0);
            cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::C);
        }
        cached.set_jit_enabled(jit);

        stepped.run(3).unwrap();
        cached.run_cached(3).unwrap();

        assert_eq!(stepped.state.reg(0), 0x8000_0000, "jit={jit}");
        assert_eq!(cached.state.reg(0), 0x8000_0000, "jit={jit}");
        assert_eq!(snapshot(&stepped, 0), snapshot(&cached, 0), "jit={jit}");
    }
}

#[test]
fn pc_as_store_value_is_not_frozen() {
    // str pc, [r0]
    let store_pc = [0xE580_F000, 0xEAFF_FFFE];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&store_pc, 0x4000);
        let mut cached = arm_cpu(&store_pc, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(0, DATA);
        }
        cached.set_jit_enabled(jit);

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.memory.peek_u32(DATA), CODE + 8, "jit={jit}");
        assert_eq!(cached.memory.peek_u32(DATA), CODE + 8, "jit={jit}");
        assert_eq!(snapshot(&stepped, 1), snapshot(&cached, 1), "jit={jit}");
    }

    // stmia r0!, {r1, pc}
    let store_multiple = [0xE8A0_8002, 0xEAFF_FFFE];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&store_multiple, 0x4000);
        let mut cached = arm_cpu(&store_multiple, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(0, DATA);
            cpu.state.set_reg(1, 0x1234);
        }
        cached.set_jit_enabled(jit);

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.memory.peek_u32(DATA), 0x1234, "jit={jit}");
        assert_eq!(stepped.memory.peek_u32(DATA + 4), CODE + 8, "jit={jit}");
        assert_eq!(cached.memory.peek_u32(DATA), 0x1234, "jit={jit}");
        assert_eq!(cached.memory.peek_u32(DATA + 4), CODE + 8, "jit={jit}");
        assert_eq!(snapshot(&stepped, 2), snapshot(&cached, 2), "jit={jit}");
    }

    // swp r0, pc, [r1]
    let swap_pc = [0xE101_009F, 0xEAFF_FFFE];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&swap_pc, 0x4000);
        let mut cached = arm_cpu(&swap_pc, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(1, DATA);
            cpu.memory.write_words(DATA, &[0x55]);
        }
        cached.set_jit_enabled(jit);

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.state.reg(0), 0x55, "jit={jit}");
        assert_eq!(cached.state.reg(0), 0x55, "jit={jit}");
        assert_eq!(stepped.memory.peek_u32(DATA), CODE + 8, "jit={jit}");
        assert_eq!(cached.memory.peek_u32(DATA), CODE + 8, "jit={jit}");
        assert_eq!(snapshot(&stepped, 1), snapshot(&cached, 1), "jit={jit}");
    }
}

#[test]
fn rrx_load_offset_keeps_carry_alive() {
    // The ldr's RRX offset depends on the C from the first adds and must be kept until a later adds overwrites it
    let program = [
        0xE096_5007, // adds r5, r6, r7 (C=1)
        0xE791_0062, // ldr r0, [r1, r2, ror #0] (RRX offset)
        0xE099_800A, // adds r8, r9, r10 (C=0)
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&program, 0x4000);
        let mut cached = arm_cpu(&program, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(1, DATA);
            cpu.state.set_reg(2, 0);
            cpu.state.set_reg(6, 0xFFFF_FFFF);
            cpu.state.set_reg(7, 1);
            cpu.state.set_reg(9, 0);
            cpu.state.set_reg(10, 0);
            cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::C);
            cpu.memory.write_words(DATA, &[0x1234]);
        }
        cached.set_jit_enabled(jit);

        stepped.run(3).unwrap();
        cached.run_cached(3).unwrap();

        // With C=1, RRX(0) = 0x8000_0000, which is out of range and reads 0; losing C would read DATA instead
        assert_eq!(stepped.state.reg(0), 0, "jit={jit}");
        assert_eq!(cached.state.reg(0), 0, "jit={jit}");
        assert_eq!(snapshot(&stepped, 1), snapshot(&cached, 1), "jit={jit}");
    }
}

#[test]
fn cached_execution_materializes_lazy_flags() {
    // subs r1, #1 (result 0, C=1 and Z=1); the block ends in a jump
    let program = [
        0xE251_1001, // subs r1, r1, #1
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        let mut cpu = arm_cpu(&program, 0x4000);
        cpu.set_jit_enabled(jit);
        cpu.state.set_reg(1, 1);
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::NZCV);

        cpu.run_cached(1).unwrap();
        if jit {
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be compiled by the JIT"
            );
        }

        assert_eq!(cpu.state.reg(1), 0, "jit={jit}");
        assert!(cpu.state.cpsr.z(), "jit={jit}");
        assert!(cpu.state.cpsr.c(), "jit={jit}");
        assert!(!cpu.state.cpsr.n(), "jit={jit}");
        assert!(
            !cpu.state.flags_are_lazy(),
            "run_cached should materialize the flags on the way out"
        );
    }
}

#[test]
fn lazy_flags_survive_block_without_flag_access() {
    // block1 produces flags and jumps to block2, which touches no flags at all
    let program = [
        0xE251_1001, // subs r1, r1, #1
        0xEAFF_FFFF, // b 0x1008
        0xE1A0_2002, // mov r2, r2
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        let mut cpu = arm_cpu(&program, 0x4000);
        cpu.set_jit_enabled(jit);
        cpu.state.set_reg(1, 1);
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::NZCV);

        cpu.run_cached(3).unwrap();
        if jit {
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be compiled by the JIT"
            );
        }

        assert_eq!(cpu.state.reg(1), 0, "jit={jit}");
        assert!(cpu.state.cpsr.z(), "jit={jit}");
        assert!(cpu.state.cpsr.c(), "jit={jit}");
        assert!(!cpu.state.flags_are_lazy(), "jit={jit}");
    }
}

#[test]
fn mrs_after_lazy_arithmetic_sees_materialized_flags() {
    let program = [
        0xE251_1001, // subs r1, r1, #1
        0xE10F_2000, // mrs r2, cpsr
        0xEAFF_FFFE, // b .
    ];
    for jit in [true, false] {
        let mut cpu = arm_cpu(&program, 0x4000);
        cpu.set_jit_enabled(jit);
        cpu.state.set_reg(1, 1);
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::NZCV);

        cpu.run_cached(2).unwrap();
        if jit {
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be compiled by the JIT"
            );
        }

        assert_eq!(cpu.state.reg(2), cpu.state.cpsr.bits(), "jit={jit}");
        assert!(cpu.state.cpsr.z(), "jit={jit}");
        assert!(cpu.state.cpsr.c(), "jit={jit}");
    }
}

#[test]
fn block_links_are_patched_for_a_loop() {
    // The static target at the tail of the subs / bne block is the block start: the self-loop back edge now takes a **direct jmp**
    // (rel32 always reaches within one buffer, so no patch slot is involved), while a cross-block edge (into the `b .` at 0x1008) still uses one.
    // The assertion here is therefore that the cross-block slot is backpatched once the loop has run to completion.
    let program = [
        0xE250_0001, // subs r0, r0, #1
        0x1AFF_FFFD, // bne 0x1000
        0xEAFF_FFFE, // b .
    ];
    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.state.set_reg(0, 1000);
    // 2000 instructions run the loop out and reach 0x1008, where the next block compiles, so the sequential exit's slot is backpatched
    cpu.run_cached(2500).unwrap();

    assert!(
        cpu.linked_site_count() > 0,
        "the cross-block slot should be linked once the loop ends"
    );
    assert_eq!(cpu.state.reg(0), 0, "the loop should run to completion");
}

#[test]
fn linked_chain_stops_at_the_instruction_budget() {
    // A link chain must not walk around `run_cached(n)`'s budget, or a `b .` becomes an infinite loop
    let program = [
        0xE250_0001, // subs r0, r0, #1
        0x1AFF_FFFD, // bne 0x1000
        0xEAFF_FFFE, // b .
    ];
    for budget in 1u64..=9 {
        let mut cpu = arm_cpu(&program, 0x4000);
        cpu.state.set_reg(0, 1_000_000);
        let executed = cpu.run_cached(budget).unwrap();
        // Granularity is the block: at least the whole budget runs, and at most one extra block (<= 32 instructions)
        assert!(
            executed >= budget && executed <= budget + 32,
            "budget={budget}, actually executed {executed}"
        );

        // Diff against the single-step path at **the same instruction count**
        let mut stepped = arm_cpu(&program, 0x4000);
        stepped.state.set_reg(0, 1_000_000);
        stepped.run(executed).unwrap();
        assert_eq!(
            snapshot(&stepped, 0),
            snapshot(&cpu, 0),
            "budget={budget}: both paths must agree once the chain ends"
        );
    }
}

#[test]
fn invalidation_does_not_leave_stale_link_targets() {
    let program = [
        0xE250_0001, // subs r0, r0, #1
        0x1AFF_FFFD, // bne 0x1000
        0xEAFF_FFFE, // b .
    ];
    let mut cpu = arm_cpu(&program, 0x4000);
    cpu.state.set_reg(0, 5);
    cpu.run_cached(20).unwrap();
    assert!(
        cpu.linked_site_count() > 0,
        "it has to actually link before invalidation means anything"
    );

    cpu.invalidate_code_cache();
    assert_eq!(
        cpu.linked_site_count(),
        0,
        "the link table must be cleared whole"
    );

    // The new code must execute once swapped in (an old link target must not stick to it)
    cpu.memory.write_words(CODE, &[0xE3A0_0007, 0xEAFF_FFFE]);
    cpu.state.set_pc(CODE);
    cpu.run_cached(1).unwrap();
    assert_eq!(cpu.state.reg(0), 7);

    // A new block has to be able to link again rather than being stuck in whatever state invalidation left
    cpu.memory.write_words(
        CODE,
        &[
            0xE250_0001, // subs r0, r0, #1
            0x1AFF_FFFD, // bne 0x1000
            0xEAFF_FFFE, // b .
        ],
    );
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, 5);
    // The code is rewritten again, so the block cache must be invalidated explicitly (the standing rule for self-modifying code)
    cpu.invalidate_code_cache();
    cpu.run_cached(20).unwrap();
    assert!(
        cpu.linked_site_count() > 0,
        "it must be able to link again after invalidation"
    );
    assert_eq!(cpu.state.reg(0), 0, "the loop should run five times");
}

#[test]
fn linking_never_crosses_frame_layouts() {
    // 32 non-accessing instructions fill one block (the translation cap) and fall through to the next, which contains an LDR,
    // so its frame carries `MEM` and a spill area. A link jump skips the target's prologue, so if this edge were backpatched
    // the target's epilogue would pop under the wrong frame layout and rsp would run away.
    //
    // The point: the program has to **come back** to the first block, or it runs only once before the target compiles and
    // the bad link is never reached at all (the first version of this case passed for exactly that reason).
    let mut words = vec![0xE1A0_0000u32; 32]; // 32 × mov r0, r0
    words.push(0xE591_0000); // 0x1080: ldr r0, [r1]
    words.push(0xEAFF_FFDD); // 0x1084: b 0x1000  -- back to the block start for another run
    let mut cpu = arm_cpu(&words, 0x4000);
    cpu.state.set_reg(1, DATA);
    cpu.memory.write_words(DATA, &[0x1234_5678]);

    let executed = cpu.run_cached(120).unwrap();

    // The same program single-stepped for the same instruction count must produce a bit-identical state
    let mut stepped = arm_cpu(&words, 0x4000);
    stepped.state.set_reg(1, DATA);
    stepped.memory.write_words(DATA, &[0x1234_5678]);
    stepped.run(executed).unwrap();

    assert!(executed >= 120, "at least the whole budget should run");
    assert!(
        cpu.cached_block_count() >= 2,
        "the cap should cut the code into two blocks"
    );
    assert_eq!(
        cpu.state.reg(0),
        0x1234_5678,
        "the LDR must keep reading the data"
    );
    assert_eq!(
        snapshot(&stepped, 0),
        snapshot(&cpu, 0),
        "a broken cross-frame link sends rsp off the rails"
    );
}

#[test]
fn decode_error_mid_block_leaves_no_partial_effects() {
    // `movs pc, lr` is an exception return (S=1 writing PC). It sits in the data-processing space, and the decoder
    // **emits an `Alu` before discovering it must refuse**. If the block path left those operations behind, an instruction
    // that should have failed would really execute once, writing r15 and NZCV, and the two paths would disagree at the failure site.
    let program = [
        0xE3A0_0007, // mov  r0, #7 (legal, and must run to completion first)
        0xE1B0_F00E, // movs pc, lr (unsupported: exception return)
    ];
    for jit in [true, false] {
        let mut stepped = arm_cpu(&program, 0x4000);
        let mut cached = arm_cpu(&program, 0x4000);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(0, 0);
            // LR's top bit is 1, so a leftover MOVS would flip N to 1
            cpu.state.set_reg(14, 0x8000_0000);
            cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::NZCV);
        }
        cached.set_jit_enabled(jit);

        assert!(
            stepped.run(2).is_err(),
            "the single-step path should report unsupported"
        );
        assert!(
            cached.run_cached(2).is_err(),
            "jit={jit}: the block path should report unsupported"
        );

        assert_eq!(
            stepped.state.reg(0),
            7,
            "jit={jit}: what came before must still execute"
        );
        assert_eq!(cached.state.reg(0), 7, "jit={jit}");
        assert_eq!(
            stepped.state.pc(),
            cached.state.pc(),
            "jit={jit}: exception address"
        );
        assert_eq!(stepped.state.r, cached.state.r, "jit={jit}");
        assert_eq!(
            stepped.state.cpsr.bits(),
            cached.state.cpsr.bits(),
            "jit={jit}: a rejected instruction must leave no flag side effect"
        );
    }
}

// ---------------------------------------------------------------------------
// Flags inside conditional-execution blocks
//
// The hot loop of `__udivsi3` repeats `cmp; subcs; orrcs` four times: all three read **the same** flags,
// and the `subcs` in the middle writes none itself. The backend has to keep the chain in the x86 flags;
// materializing mid-chain would cost a dozen host instructions per guard.
// ---------------------------------------------------------------------------

/// One `cmp` feeds two guards with a flag-free subtraction in between, and the flags must not be destroyed.
#[test]
fn conditional_chain_shares_one_set_of_flags() {
    // The crux is that `subcs r2, r1, r0` subtracts **backwards**: were it taken as the source of new flags,
    // C would flip from 1 to 0 (5-9 borrows) and the second guard would skip when it should not.
    // Subtracting in the same direction makes both implementations agree and hides the bug — that lesson was learned the hard way.
    let program = [
        0xE150_0001, // cmp   r0, r1
        0x2041_2000, // subcs r2, r1, r0
        0x2383_3001, // orrcs r3, r3, #1
        0xEAFF_FFFE, // b .
    ];
    // With C set (r0 > r1, no borrow) both execute; otherwise neither does
    for (r0, r1, expect) in [(9u32, 5u32, (0xFFFF_FFFCu32, 1u32)), (3, 5, (0, 0))] {
        let mut runs = Vec::new();
        for jit in [false, true] {
            let mut cpu = arm_cpu(&program, 0x4000);
            cpu.state.set_reg(0, r0);
            cpu.state.set_reg(1, r1);
            cpu.set_jit_enabled(jit);
            if jit {
                cpu.run_cached(3).unwrap();
                assert!(
                    cpu.compiled_block_count() > 0,
                    "this chain has to really take the JIT path"
                );
            } else {
                cpu.run(3).unwrap();
            }
            runs.push((cpu.state.reg(2), cpu.state.reg(3), cpu.state.cpsr.bits()));
        }
        assert_eq!(runs[0], runs[1], "r0={r0} r1={r1}: the two paths disagree");
        assert_eq!(
            (runs[1].0, runs[1].1),
            expect,
            "r0={r0} r1={r1}: wrong result"
        );
    }
}

/// Flags written conditionally with no reader must be stripped, while ones that really are read must stay.
#[test]
fn guarded_flag_write_is_kept_only_when_someone_reads_it() {
    use rynarmic::block::Translation;

    fn translate_program(program: &[u32]) -> rynarmic::block::BasicBlock {
        let mut memory = SimpleMemory::new(0, 0x4000);
        memory.write_words(CODE, program);
        memory.write_words(CODE + 4 * program.len() as u32, &[0xEAFF_FFFE]);
        match rynarmic::block::translate(&mut memory, CODE, false).unwrap() {
            Translation::Block(block) => block,
            other => panic!("should form a block: {other:?}"),
        }
    }

    fn guarded_sub_set_flags(program: &[u32]) -> bool {
        translate_program(program)
            .ops
            .iter()
            .find_map(|op| match op {
                // The guarded conditional subtraction (with S)
                rynarmic::IrOp::Alu {
                    op: rynarmic::AluOp::Sub,
                    dst: Some(0),
                    set_flags,
                    ..
                } => Some(*set_flags),
                _ => None,
            })
            .expect("the block should hold one conditional subtraction")
    }

    // A full write follows subscs immediately, so nobody reads that flag write and it should be stripped
    let dead = [0xE150_0001, 0x2050_0001, 0xE350_0000];
    assert!(
        !guarded_sub_set_flags(&dead),
        "nothing reads the flags between the merge and the next full write, so the write should go"
    );

    // `orrcs` reads C after subscs, so that write is live and has to stay
    let live = [0xE150_0001, 0x2050_0001, 0x2182_2003];
    assert!(
        guarded_sub_set_flags(&live),
        "orrcs reads the carry subscs produced, so the flag write must stay"
    );

    // The liveness table has to line up with the IR item by item. Here the first `cmp` has no reader and is dropped whole, shrinking the IR —
    // and if the table did not shrink with it, the backend would read the liveness of the wrong position when deciding whether the flags are dead
    // (a differential test caught this once, and the whole CPSR was wrong as a result).
    let dropped = [
        0xE150_0001, // cmp r0, r1   -- overwritten by the next instruction, so dropped whole
        0xE150_0002, // cmp r0, r2
        0xE350_0000, // cmp r0, #0
        0x2050_0001, // subscs r0, r0, r1
    ];
    let block = translate_program(&dropped);
    assert_eq!(
        block.flag_live_at.len(),
        block.ops.len() + 1,
        "the liveness table must line up with the IR item by item"
    );
}

/// A shifted conditional subtraction: the shift destroys the x86 flags, while the next guard reads that same set.
///
/// In that situation the backend captures the condition into a byte with `setcc` (`FlagMode::Captured`)
/// and later same-condition guards read it. This case watches that path specifically:
/// get the captured polarity backwards, or let the guard read destroyed flags, and `r3` comes out wrong.
#[test]
fn conditional_chain_with_shifted_operand_captures_flags() {
    // The trailing `cmp` is required: it makes the flags "die" at the chain's end, which is what admits the capture chain
    // (in real code this is `__udivsi3`'s `cmp; subcs; orrcs; cmp`).
    let program = [
        0xE150_0001, // cmp   r0, r1
        0x2041_20A0, // subcs r2, r1, r0, lsr #1
        0x2383_3001, // orrcs r3, r3, #1
        0xE150_0001, // cmp   r0, r1   -- the flags are overwritten here
        0xEAFF_FFFE, // b .
    ];
    // r0 > r1 gives C=1: the subtraction runs (r2 = r1 - r0/2) and so does orrcs (r3 |= 1)
    for (r0, r1, expect_r2, expect_r3) in [
        (6u32, 4u32, 4u32 - 3, 1u32), // 6>>1 = 3 → r2 = 1
        (4, 6, 0, 0),                 // C=0: neither executes
    ] {
        let mut runs = Vec::new();
        for jit in [false, true] {
            let mut cpu = arm_cpu(&program, 0x4000);
            cpu.state.set_reg(0, r0);
            cpu.state.set_reg(1, r1);
            cpu.state.set_reg(2, 0);
            cpu.state.set_reg(3, 0);
            cpu.set_jit_enabled(jit);
            if jit {
                cpu.run_cached(3).unwrap();
                assert!(
                    cpu.compiled_block_count() > 0,
                    "this has to really take the JIT path"
                );
            } else {
                cpu.run(3).unwrap();
            }
            runs.push((cpu.state.reg(2), cpu.state.reg(3), cpu.state.cpsr.bits()));
        }
        assert_eq!(runs[0], runs[1], "r0={r0} r1={r1}: the two paths disagree");
        assert_eq!(
            (runs[1].0, runs[1].1),
            (expect_r2, expect_r3),
            "r0={r0} r1={r1}"
        );
    }
}
