//! Interpreter throughput benchmark, providing the baseline that block-level IR and its optimizations are measured against.
//!
//! Usage: `cargo run --release --example bench`

use std::time::Instant;

use rynarmic::{Cpu, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const SPIN: u32 = 0x2000;

fn bench_arm(iterations: u32, cached: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE250_0001, // subs r0, r0, #1
            0x1AFF_FFFD, // bne  0x1000
            0xEAFF_FFFE, // b .
        ],
    );
    memory.write_words(SPIN, &[0xEAFF_FFFE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, iterations);

    // Two instructions per iteration, plus one final `b .`
    let budget = (iterations as u64) * 2 + 1;
    let start = Instant::now();
    if cached {
        cpu.run_cached(budget)
            .expect("the benchmark should raise no exception");
    } else {
        cpu.run(budget)
            .expect("the benchmark should raise no exception");
    }
    let elapsed = start.elapsed().as_secs_f64();
    budget as f64 / elapsed / 1_000_000.0
}

fn bench_thumb(iterations: u32, cached: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x3901, // subs r1, #1
            0xD1FD, // bne 0x1000
            0xE7FE, // b .
        ],
    );
    memory.write_thumb(SPIN, &[0xE7FE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, iterations);

    let budget = (iterations as u64) * 2 + 1;
    let start = Instant::now();
    if cached {
        cpu.run_cached(budget)
            .expect("the benchmark should raise no exception");
    } else {
        cpu.run(budget)
            .expect("the benchmark should raise no exception");
    }
    let elapsed = start.elapsed().as_secs_f64();
    budget as f64 / elapsed / 1_000_000.0
}

/// A conditionally executed ARM loop: `cmp`, `moveq`/`addne`, `subs`, `bne`.
///
/// Such code is very common in A32 (compilers use conditional execution to remove short branches). If conditional instructions could not go into a block,
/// every one of them would force a fallback to single-stepping and blocks would be cut down to one to three instructions.
fn bench_arm_predicated(iterations: u32, cached: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(
        CODE,
        &[
            0xE350_0000, // cmp   r0, #0
            0x03A0_1001, // moveq r1, #1
            0x1281_1002, // addne r1, r1, #2
            0xE250_0001, // subs  r0, r0, #1
            0x1AFF_FFFB, // bne   CODE
            0xEAFF_FFFE, // b .
        ],
    );
    memory.write_words(SPIN, &[0xEAFF_FFFE]);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, iterations);
    cpu.state.cpsr.set_z(true);

    let budget = (iterations as u64) * 4 + 1;
    let start = Instant::now();
    if cached {
        cpu.run_cached(budget)
            .expect("the benchmark should raise no exception");
    } else {
        cpu.run(budget)
            .expect("the benchmark should raise no exception");
    }
    budget as f64 / start.elapsed().as_secs_f64() / 1_000_000.0
}

/// A loop closer to real Thumb code: a PC-relative constant load, arithmetic, and a conditional branch.
fn bench_thumb_literal(iterations: u32, cached: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(
        CODE,
        &[
            0x480F, // 1000: ldr r0, [pc, #0x3C]  -> 0x1040
            0x3101, // 1002: adds r1, #1
            0x3A01, // 1004: subs r2, #1
            0xD1FB, // 1006: bne 0x1000
            0xE7FE, // 1008: b .
        ],
    );
    memory.write_words(0x1040, &[0x1234_5678]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(2, iterations);

    let budget = (iterations as u64) * 4 + 1;
    let start = Instant::now();
    if cached {
        cpu.run_cached(budget)
            .expect("the benchmark should raise no exception");
    } else {
        cpu.run(budget)
            .expect("the benchmark should raise no exception");
    }
    budget as f64 / start.elapsed().as_secs_f64() / 1_000_000.0
}

/// Best of several runs, because this measurement is noisy in one direction only.
///
/// Repeating the same binary on the machine this was written on moved a row by up to 49% (`Thumb
/// with PC-relative loads` ranged 1926-2875 MIPS over three runs) — thermal drift on a laptop, not
/// code. Noise only ever makes a run *slower*, so the minimum is the closest thing to the number the
/// code owns, and it is the only way a comparison against a stored baseline means anything.
/// `compiled_bench` already did this; this example did not, which is why its first recorded table and
/// its next disagreed by more than any code change could explain.
const ROUNDS: usize = 3;

fn best(mut run: impl FnMut() -> f64) -> f64 {
    (0..ROUNDS).map(|_| run()).fold(f64::MIN, f64::max)
}

fn main() {
    let iterations = 5_000_000;

    // Warm up first, so that the first allocation and a cold instruction cache stay out of the numbers
    bench_arm(100_000, false);
    bench_thumb(100_000, false);
    bench_arm(100_000, true);
    bench_thumb(100_000, true);

    bench_thumb_literal(100_000, false);
    bench_thumb_literal(100_000, true);
    bench_arm_predicated(100_000, false);
    bench_arm_predicated(100_000, true);

    let arm_step = best(|| bench_arm(iterations, false));
    let thumb_step = best(|| bench_thumb(iterations, false));
    let arm_cached = best(|| bench_arm(iterations, true));
    let thumb_cached = best(|| bench_thumb(iterations, true));

    println!("iterations: {iterations} (two instructions each), best of {ROUNDS} runs\n");
    println!("A32   interpreted: {arm_step:7.1} MIPS");
    println!(
        "A32   block cache: {arm_cached:7.1} MIPS  ({:.2}x)",
        arm_cached / arm_step
    );
    println!("Thumb interpreted: {thumb_step:7.1} MIPS");
    println!(
        "Thumb block cache: {thumb_cached:7.1} MIPS  ({:.2}x)",
        thumb_cached / thumb_step
    );
    let lit_step = best(|| bench_thumb_literal(iterations, false));
    let lit_cached = best(|| bench_thumb_literal(iterations, true));
    println!();
    let pred_step = best(|| bench_arm_predicated(iterations, false));
    let pred_cached = best(|| bench_arm_predicated(iterations, true));
    println!("A32 with conditional execution (four per round):");
    println!("  interpreted: {pred_step:7.1} MIPS");
    println!(
        "  block cache: {pred_cached:7.1} MIPS  ({:.2}x)",
        pred_cached / pred_step
    );
    println!();
    println!("Thumb with PC-relative loads (four per round):");
    println!("  interpreted: {lit_step:7.1} MIPS");
    println!(
        "  block cache: {lit_cached:7.1} MIPS  ({:.2}x)",
        lit_cached / lit_step
    );
    println!();
    println!(
        "Note: the block cache removes repeated decoding and per-instruction PC dispatch; translating blocks to machine code adds another order of magnitude."
    );
    println!(
        "      Accesses go through host callbacks and registers stay in memory; see M4.2/M4.3 in the plan."
    );
}
