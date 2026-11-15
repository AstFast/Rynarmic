//! A flag regression over the three exits of a self-looping branch (`subs ; bne .`).
//!
//! Storing the flags moved from the hot path to the two cold exits (the sequential one and the taken one where the budget runs out),
//! but that store once sat after instructions that destroy the x86 flags (`test T3,T3` clears CF and OF,
//! `cmp budget,0` rewrites CF), so C and V came from the condition code or the budget rather than from the arithmetic result.
//! The three cases cover the sequential exit, the sequential exit with a borrow, and the taken cold exit.

use rynarmic::{Cpsr, Cpu, memory::SimpleMemory};

const CODE: u32 = 0x1000;

fn cpu_with(words: &[u32], r0: u32) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, r0);
    // Not Thumb, SVC mode, with C/Z/N/V all clear
    cpu.state.cpsr = Cpsr::from_bits(0x13);
    cpu
}

fn check(name: &str, r0: u32, budget: u64) {
    check_with(name, r0, budget, 0x1AFF_FFFD); // bne 0x1000
}

fn check_with(name: &str, r0: u32, budget: u64, branch: u32) {
    let program = [
        0xE250_0001, // subs r0, r0, #1
        branch,      // jump back to the block start on the condition (a self-loop)
    ];
    let mut stepped = cpu_with(&program, r0);
    stepped.run(budget).unwrap();
    let mut cached = cpu_with(&program, r0);
    let executed = cached.run_cached(budget).unwrap();

    let s = stepped.state.cpsr.bits();
    let c = cached.state.cpsr.bits();
    println!(
        "{name}: r0={r0:#x} budget={budget} single-stepped={} blocks executed={executed}\n  \
         single-step cpsr={s:#010x} (NZCV={:04b}) block cpsr={c:#010x} (NZCV={:04b})",
        stepped.state.pc(),
        s >> 28,
        c >> 28
    );
    assert_eq!(
        (cached.state.r[0], c),
        (stepped.state.r[0], s),
        "{name}: the block path disagrees with the single-step path (block ran {executed})"
    );
}

#[test]
fn self_loop_cold_exits_keep_flags() {
    // A: bne does not hold, so the sequential exit. subs 1-1 has no borrow, so C=1.
    check("sequential exit", 1, 2);
    // B: the budget runs out exactly on the taken edge, so the taken cold exit. subs 0-1 borrows, so C=0 and N=1.
    check("taken cold exit", 0, 1);
    // C: budget to spare, so the hot path loops and exits through the next block.
    check("hot loop", 5, 7);
}

/// The sequential exit with a borrow: `subs r1, r0, #1` (0-1 -> 0xFFFFFFFF, C=0 and N=1)
/// and `beq` (Z=0) fails, so it takes the sequential exit. The true C is 0, and if the descriptor came from
/// the CF cleared by `test T3,T3` it would be inverted by sub_like into C=1.
#[test]
fn self_loop_fallthrough_with_borrow() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE250_1001, // subs r1, r0, #1
            0x0AFF_FFFD, // beq 0x1000 (Z=0, so not taken: the sequential exit)
        ],
    );
    let mut stepped = Cpu::new(memory);
    stepped.state.set_pc(CODE);
    stepped.state.cpsr = Cpsr::from_bits(0x13);
    stepped.run(2).unwrap();

    let mut memory2 = SimpleMemory::new(0, 0x4000);
    memory2.write_words(CODE, &[0xE250_1001, 0x0AFF_FFFD]);
    let mut cached = Cpu::new(memory2);
    cached.state.set_pc(CODE);
    cached.state.cpsr = Cpsr::from_bits(0x13);
    cached.run_cached(2).unwrap();

    println!(
        "sequential exit with borrow: single-step cpsr={:#010x} (NZCV={:04b}) block cpsr={:#010x} (NZCV={:04b})",
        stepped.state.cpsr.bits(),
        stepped.state.cpsr.bits() >> 28,
        cached.state.cpsr.bits(),
        cached.state.cpsr.bits() >> 28
    );
    assert_eq!(cached.state.cpsr.bits(), stepped.state.cpsr.bits());
}
