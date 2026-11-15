//! Throughput of realistic-shape kernels: single-stepped against the block cache plus JIT.
//!
//! Unlike `bench.rs`, whose four loops are clean two-to-four-instruction cycles that say nothing about the shape of real code,
//! these three are hand-assembled kernels with longer blocks and a mix of branches and accesses (see `tests/support/kernels.rs`).
//! They double as correctness cases (`tests/real_workload.rs` asserts on expected values), so none of them is fast and wrong.
//!
//! Usage: `cargo run --release --example real_code`
//!
//! Note: these are not CoreMark, only kernels whose **shape** is close to compiler output.
//! Any "real code runs at x MIPS" conclusion drawn from them is a lower bound only.

#[path = "../tests/support/kernels.rs"]
mod kernels;

use std::time::Instant;

use rynarmic::{Cpu, memory::SimpleMemory};

/// How many instructions each kernel runs (long enough to be stable, short enough not to take forever).
const TARGET_INSTRUCTIONS: u64 = 4_000_000;

/// Rounds per arm, taking the better value (thermal drift alone can reach 30%, so one reading is not trustworthy).
const ROUNDS: usize = 2;

type Reset<'a> = &'a mut dyn FnMut(&mut Cpu<SimpleMemory>);

fn new_cpu(program: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(kernels::CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(kernels::CODE);
    cpu
}

/// Runs one kernel repeatedly and returns MIPS.
fn measure(program: &[u32], per_run: u64, cached: bool, reset: Reset<'_>) -> f64 {
    let mut cpu = new_cpu(program);
    cpu.set_jit_enabled(cached);

    let mut executed = 0u64;
    let start = Instant::now();
    while executed < TARGET_INSTRUCTIONS {
        cpu.state.set_pc(kernels::CODE);
        reset(&mut cpu);
        executed += if cached {
            cpu.run_cached(per_run)
                .expect("the kernel should raise no exception")
        } else {
            cpu.run(per_run)
                .expect("the kernel should raise no exception")
        };
    }
    let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
    executed as f64 / elapsed / 1_000_000.0
}

fn report(name: &str, program: &[u32], per_run: u64, reset: Reset<'_>) {
    let mut interp = 0.0f64;
    let mut jit = 0.0f64;
    for _ in 0..ROUNDS {
        interp = interp.max(measure(program, per_run, false, reset));
        jit = jit.max(measure(program, per_run, true, reset));
    }
    println!(
        "{name:<22} step {interp:>7.1}   block+JIT {jit:>7.1} MIPS   ({:.1}x)",
        jit / interp
    );
}

fn main() {
    let udiv = kernels::udiv_program();
    let memcpy = kernels::memcpy_program();
    let call = kernels::call_program_with_function();

    println!(
        "each kernel runs {TARGET_INSTRUCTIONS} instructions; best of {ROUNDS} rounds per arm.\n"
    );

    report(
        "udiv (branch-dense)",
        &udiv,
        kernels::UDIV_INSTRUCTIONS,
        &mut |cpu| {
            cpu.state.set_reg(0, 0xFFFF_FFFF);
            cpu.state.set_reg(1, 7);
        },
    );

    report("memcpy (mixed accesses)", &memcpy, 160, &mut |cpu| {
        cpu.state.set_reg(0, kernels::DST);
        cpu.state.set_reg(1, kernels::SRC);
        cpu.state.set_reg(2, 64);
    });

    report(
        "call (bl / bx)",
        &call,
        kernels::CALL_INSTRUCTIONS,
        &mut |_| {},
    );

    println!(
        "\nNote: these three are hand-assembled kernels shaped like compiler output (branch-dense, blocks of 9 to 14 instructions,\n\
         mixing accesses and arithmetic); they are not CoreMark, so conclusions drawn from them are lower bounds only."
    );
}
