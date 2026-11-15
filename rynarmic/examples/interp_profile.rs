//! A cost breakdown of the interpreter and the translator.
//!
//! The key method: **difference** two measurements, a two-instruction body and a thirty-instruction one, to separate the fixed
//! per-block entry and exit cost (cache lookup, dispatch, PC sync) from the marginal cost of one IR operation;
//! MIPS alone mixes the two and says nothing about which end to optimize.
use rynarmic::block::translate;
use rynarmic::memory::SimpleMemory;
use rynarmic::{Cpsr, Cpu, Memory};
use std::time::Instant;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
/// Rounds per configuration (the best is taken).
const ROUNDS: usize = 5;

/// The body is `body` copies of `instr` followed by a `b` back to the block start (a self-loop), forming a cacheable block.
fn build(body: usize, instr: u32) -> Cpu<SimpleMemory> {
    let mut words = vec![instr; body];
    let offset = -((body as i32) + 2);
    words.push(0xEA00_0000 | (offset as u32 & 0x00FF_FFFF));
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(CODE, &words);
    memory.write_u32(DATA, 0x1234_5678);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13);
    cpu.state.set_reg(1, DATA);
    cpu
}

/// Mean nanoseconds per **block** (block-level interpretation with the JIT off). Best of several rounds, because the measure is
/// extremely sensitive to machine noise (scheduling jitter, clock changes), with a single round's spread reaching 30%.
fn ns_per_block(body: usize, instr: u32, instructions: u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let mut cpu = build(body, instr);
        cpu.set_jit_enabled(false);
        // A short warm-up first, so the first translation is not folded into the measurement
        cpu.run_cached(1000).unwrap();
        let t = Instant::now();
        cpu.run_cached(instructions).unwrap();
        let elapsed = t.elapsed().as_secs_f64();
        // `run_cached` accounts by block, so the count may exceed by one block; rounding to whole blocks is enough
        let blocks = (instructions / (body as u64 + 1)) as f64;
        best = best.min(elapsed / blocks * 1e9);
    }
    best
}

fn main() {
    let n = 40_000_000u64;

    let mut best_step = f64::MAX;
    for _ in 0..ROUNDS {
        let mut cpu = build(2, 0xE280_0001);
        let t = Instant::now();
        cpu.run(n).unwrap();
        best_step = best_step.min(t.elapsed().as_secs_f64());
    }
    let step = n as f64 / best_step / 1e6;

    let small = 2;
    let large = 30;
    let block_small = ns_per_block(small, 0xE280_0001, n);
    let block_large = ns_per_block(large, 0xE280_0001, n);
    let ns_add = (block_large - block_small) / (large - small) as f64;
    let fixed = block_small - small as f64 * ns_add;

    // Accesses: through callbacks (SimpleMemory has a window, but the interpreter never inlines and always calls back)
    let load_small = ns_per_block(small, 0xE591_0000, n); // ldr r0, [r1]
    let load_large = ns_per_block(large, 0xE591_0000, n);
    let ns_ldr = (load_large - load_small) / (large - small) as f64;

    // Cold translation cost: a different pc each time, to defeat every cache
    let mut memory = SimpleMemory::new(0, 0x40000);
    for i in 0..2048u32 {
        memory.write_words(CODE + i * 8, &[0xE280_0001, 0xEAFF_FFFE]);
    }
    let reps = 200_000u64;
    let t = Instant::now();
    for i in 0..reps {
        let pc = CODE + (i as u32 & 0x7FF) * 8;
        let _ = translate(&mut memory, pc, false).unwrap();
    }
    let ns = t.elapsed().as_secs_f64() / reps as f64 * 1e9;

    println!("single step       : {step:7.1} MIPS");
    println!(
        "block interpreted (2 insns)  : {:7.1} MIPS",
        (small + 1) as f64 / block_small * 1e3
    );
    println!(
        "block interpreted (30 insns) : {:7.1} MIPS",
        (large + 1) as f64 / block_large * 1e3
    );
    println!("fixed cost per block : {fixed:7.1} ns (cache lookup + dispatch + PC sync)");
    println!("marginal cost, add   : {ns_add:7.2} ns/insn");
    println!("marginal cost, ldr   : {ns_ldr:7.2} ns/insn");
    println!("translate            : {ns:7.1} ns/block");
}
