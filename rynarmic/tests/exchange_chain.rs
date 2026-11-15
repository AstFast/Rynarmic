//! The inline-cache direct link for exchange branches (`mov pc, lr`, `bx lr`): a correctness regression.
//!
//! The machine code **jumps straight into the target's body** on a hit, skipping prologue and epilogue, so three things are pinned here:
//! 1. results after linking are identical to the single-step path;
//! 2. a different target at the same exchange (a cache miss) is still correct;
//! 3. after the guest rewrites code and `invalidate_code_cache()` runs, no old machine code executes.

use rynarmic::memory::{Memory, SimpleMemory};
use rynarmic::{Cpsr, Cpu};

const CODE: u32 = 0x1000;

fn cpu_with(words: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13);
    cpu
}

/// `mov r1,#20 ; loop: add r0,r0,#1 ; subs r1,r1,#1 ; movne pc,lr ; b .`
/// An exchange every three instructions, 19 hits on the inline cache, ending in a `b .` spin.
fn loop_words() -> Vec<u32> {
    vec![
        0xE3A0_1014, // mov r1, #20
        0xE280_0001, // loop: add r0, r0, #1
        0xE251_1001, // subs r1, r1, #1
        0x11A0_F00E, // movne pc, lr
        0xEAFF_FFFE, // b .
    ]
}

fn run_both(
    words: &[u32],
    setup: impl Fn(&mut Cpu<SimpleMemory>),
) -> (Cpu<SimpleMemory>, Cpu<SimpleMemory>) {
    let mut stepped = cpu_with(words);
    setup(&mut stepped);
    stepped.run(200).unwrap();

    let mut cached = cpu_with(words);
    setup(&mut cached);
    cached.run_cached(200).unwrap();
    assert!(
        cached.compiled_block_count() >= 2,
        "both blocks should be compiled"
    );
    (stepped, cached)
}

#[test]
fn exchange_chain_matches_step_path() {
    let words = loop_words();
    let (stepped, cached) = run_both(&words, |cpu| cpu.state.set_reg(14, CODE + 4));
    assert_eq!(stepped.state.r, cached.state.r, "the registers must agree");
    assert_eq!(
        stepped.state.cpsr.bits(),
        cached.state.cpsr.bits(),
        "the flags must agree"
    );
    assert_eq!(stepped.state.pc(), cached.state.pc());
    assert_eq!(
        cached.state.reg(0),
        20,
        "the loop should complete 20 rounds"
    );
    assert_eq!(stepped.state.reg(0), 20);
}

#[test]
fn exchange_chain_refills_when_target_changes() {
    // One Cpu runs a round first (the inline cache now holds CODE+4), then re-enters **in the middle of a block**,
    // so that the same exchange sees a new target. Correctness means "bit-identical to the single-step path", with no hard-coded expectation.
    let words = loop_words();
    let scenario = |cached: bool| {
        let mut cpu = cpu_with(&words);
        cpu.state.set_reg(14, CODE + 4);
        let run = |cpu: &mut Cpu<SimpleMemory>| {
            if cached {
                cpu.run_cached(200).unwrap();
            } else {
                cpu.run(200).unwrap();
            }
        };
        run(&mut cpu);
        // Re-enter at `subs` (this block excludes `add`) and reset r1 to six rounds
        cpu.state.set_pc(CODE + 8);
        cpu.state.set_reg(1, 6);
        cpu.state.set_reg(14, CODE + 8);
        cpu.state.set_reg(0, 0);
        run(&mut cpu);
        (cpu.state.r, cpu.state.cpsr.bits(), cpu.state.pc())
    };
    let (cached_r, cached_cpsr, cached_pc) = scenario(true);
    let (step_r, step_cpsr, step_pc) = scenario(false);
    assert_eq!(
        cached_r, step_r,
        "after a target change the registers must match the single-step path"
    );
    assert_eq!(cached_cpsr, step_cpsr, "the flags must agree");
    assert_eq!(cached_pc, step_pc);
}

#[test]
fn invalidating_cache_after_self_modification_runs_new_code() {
    let words = loop_words();
    let mut cpu = cpu_with(&words);
    cpu.state.set_reg(14, CODE + 4);
    cpu.run_cached(200).unwrap();
    assert_eq!(cpu.state.reg(0), 20);

    // The guest rewrites `add r0,r0,#1` as `add r0,r0,#3`
    cpu.memory.write_u32(CODE + 4, 0xE280_0003);
    cpu.invalidate_code_cache();
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, 0);
    cpu.state.set_reg(1, 20);
    cpu.run_cached(200).unwrap();
    assert_eq!(
        cpu.state.reg(0),
        60,
        "the rewritten code must run (+3 per round)"
    );
}
