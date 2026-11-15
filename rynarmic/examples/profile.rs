//! Breaks down the fixed cost of a basic block: stuff different numbers of harmless instructions into one loop and watch throughput.
//!
//! Usage: `cargo run --release --example profile`
//!
//! A linear fit of `time per round = fixed cost + N x per-instruction cost` shows whether to optimize the block's entry and exit
//! or the instructions inside it.

use std::time::Instant;

use rynarmic::{Cpu, memory::SimpleMemory};

const CODE: u32 = 0x1000;

/// The loop body: N copies of `mov r1, r1` (no flags, no access), then `subs r0, r0, #1` and `bne`.
fn loop_program(n: usize) -> Vec<u32> {
    let mut program = vec![0xE1A0_1001u32; n]; // mov r1, r1
    program.push(0xE250_0001); // subs r0, r0, #1
    // bne back to the loop head: target = CODE, current address = CODE + 4n + 4
    let here = CODE + 4 * (n as u32) + 4;
    let offset = (CODE as i64 - (here as i64 + 8)) / 4;
    program.push(0x1A00_0000 | (offset as u32 & 0x00FF_FFFF));
    program
}

fn measure(program: &[u32], iterations: u32, jit: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(0, iterations);
    cpu.set_jit_enabled(jit);

    let per_round = program.len() as u64;
    let budget = iterations as u64 * per_round / per_round * per_round;
    let start = Instant::now();
    cpu.run_cached(budget)
        .expect("the benchmark should raise no exception");
    start.elapsed().as_secs_f64()
}

fn main() {
    let iterations = 20_000_000u32;

    println!("each round is N movs plus subs plus bne");
    println!(
        "{:>3}  {:>9}  {:>9}  {:>9}",
        "N", "rounds/s (10k)", "ns/round", "ns/insn"
    );
    let mut points: Vec<(f64, f64)> = Vec::new();
    for n in [0usize, 1, 2, 4, 8, 16] {
        let program = loop_program(n);
        // Warm up
        measure(&program, 100_000, true);
        let elapsed = measure(&program, iterations, true);
        let ns_per_round = elapsed * 1e9 / iterations as f64;
        let instructions = (n + 2) as f64;
        println!(
            "{n:>3}  {:>9.1}  {ns_per_round:>9.2}  {:>9.2}",
            iterations as f64 / elapsed / 1e4,
            ns_per_round / instructions
        );
        points.push((instructions, ns_per_round));
    }

    // Least-squares fit of ns = f + c * instructions
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let intercept = (sy - slope * sx) / n;
    println!(
        "\nfit: per round, about {intercept:.2} ns fixed cost + {slope:.2} ns x instruction count"
    );
    println!(
        "-> if the fixed cost dominates, optimize the block entry and exit; otherwise the instructions inside"
    );

    println!("\ncontrol: JIT off (pure interpreter)");
    let program = loop_program(4);
    let elapsed = measure(&program, iterations, false);
    println!(
        "  N=4 interpreted: {:.1} x10k rounds/s, {:.2} ns/round",
        iterations as f64 / elapsed / 1e4,
        elapsed * 1e9 / iterations as f64
    );
}
