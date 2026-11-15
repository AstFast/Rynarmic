//! Measured throughput of genuinely compiled ARM kernels (`arm-none-eabi-gcc -O2`).
//!
//! Usage: `cargo run --release --example compiled_bench`
//! Without a toolchain it says so and exits (`build.rs` does not fail).
//!
//! Unlike `real_code.rs`, whose kernels are hand-assembled and merely close in shape, this is machine code a compiler really produced:
//! the block boundaries, the register allocation and the `__aeabi_uidiv` calls are all real.

include!(concat!(env!("OUT_DIR"), "/compiled_bench.rs"));

use std::time::{Duration, Instant};

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

const MEMORY_SIZE: usize = 0x40_0000;
const STACK: u32 = 0x3F_000;
const DST: u32 = 0x4_000;
const SRC: u32 = 0x2_0000;
const COPY_WORDS: u32 = 4096;
const LIMIT: u64 = 40_000_000;
const ROUNDS: usize = 3;

fn new_cpu(jit: bool) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, MEMORY_SIZE);
    memory.write_bytes(BASE, BLOB);
    let mut cpu = Cpu::new(memory);
    cpu.set_jit_enabled(jit);
    // Inlined accesses can be switched off inside one binary, so the A/B needs no separate process and is immune to clock changes
    if std::env::var_os("RYNARMIC_NO_FASTMEM").is_some() {
        cpu.set_fastmem_enabled(false);
    }
    cpu
}

fn enter(cpu: &mut Cpu<SimpleMemory>, which: u32, args: [u32; 3]) {
    cpu.state.set_pc(ENTRY);
    cpu.state.set_reg(13, STACK);
    cpu.state.set_reg(0, which);
    cpu.state.set_reg(1, args[0]);
    cpu.state.set_reg(2, args[1]);
    cpu.state.set_reg(3, args[2]);
}

/// Runs once and returns `(elapsed, instruction count reported by the JIT path)`.
///
/// The single-step path's count is not measured separately: for one piece of code and one input the executed count is deterministic,
/// so the JIT's number is used for both arms, which is what makes them comparable.
fn once(cpu: &mut Cpu<SimpleMemory>, which: u32, args: [u32; 3]) -> (Duration, u64) {
    enter(cpu, which, args);
    let start = Instant::now();
    let (elapsed, count) = if cpu.jit_enabled() {
        let (executed, result) = cpu.run_cached_counted(LIMIT);
        assert!(result.is_err(), "the kernel should end with a swi");
        (start.elapsed(), executed)
    } else {
        // `run` returns Err at the swi, which is exactly "it finished"
        let _ = cpu.run(LIMIT);
        (start.elapsed(), 0)
    };
    (elapsed, count)
}

fn report(name: &str, which: u32, args: [u32; 3], prep: impl Fn(&mut Cpu<SimpleMemory>)) {
    // The JIT arm: one warm-up (to get the blocks compiled), then the best of several rounds
    let mut cpu = new_cpu(true);
    prep(&mut cpu);
    let (_, count) = once(&mut cpu, which, args);
    let blocks = cpu.compiled_block_count();
    // How many blocks translated but could not be compiled and fell back to the interpreter: above zero the backend has a coverage gap and throughput collapses
    let fallback = cpu.cached_block_count().saturating_sub(blocks);
    let mut best_jit = Duration::MAX;
    for _ in 0..ROUNDS {
        best_jit = best_jit.min(once(&mut cpu, which, args).0);
    }

    // The single-step arm
    let mut cpu = new_cpu(false);
    prep(&mut cpu);
    let _ = once(&mut cpu, which, args);
    let mut best_interp = Duration::MAX;
    for _ in 0..ROUNDS {
        best_interp = best_interp.min(once(&mut cpu, which, args).0);
    }

    let to_mips = |best: Duration| count as f64 / best.as_secs_f64() / 1_000_000.0;
    let (jit, interp) = (to_mips(best_jit), to_mips(best_interp));
    println!(
        "{name:<20} {count:>7} insns   step {interp:>7.1}   block+JIT {jit:>7.1} MIPS   ({:.1}x)   [{blocks} blocks{fallback}]",
        jit / interp,
        fallback = if fallback == 0 {
            String::new()
        } else {
            format!(" / {fallback} interpreted")
        }
    );
}

fn main() {
    if !AVAILABLE {
        println!(
            "no ARM cross toolchain, skipping. Set RYNARMIC_ARM_GCC to point at arm-none-eabi-gcc."
        );
        return;
    }

    println!(
        "bare-metal image of {} bytes (entry={ENTRY:#X}); best of {ROUNDS} rounds per arm.\n",
        BLOB.len()
    );

    report("div (__aeabi_uidiv)", 0, [1000, 0xDEAD_BEEF, 0], |_| {});

    report("copy (4096 words)", 1, [DST, SRC, COPY_WORDS], |cpu| {
        for i in 0..COPY_WORDS {
            cpu.memory
                .write_u32(SRC + i * 4, i.wrapping_mul(2_654_435_761));
        }
    });

    report("call (2000 calls)", 2, [2000, 0x1234_5678, 0], |_| {});

    // The acceptance measure for block-local value propagation plus dead-write elimination: how many IR operations it removed
    let (blocks, before, after) = rynarmic::ir::optimize_totals();
    if blocks > 0 {
        let ratio = 100.0 * before.saturating_sub(after) as f64 / before as f64;
        println!(
            "\nIR optimization: {blocks} block translations, IR operations {before} -> {after} (-{ratio:.1}%); \
             levels: RYNARMIC_IR_OPT=off|prop|dce|full (full by default)"
        );
    }

    println!(
        "\nNote: these three are real compiled output (-O2), more trustworthy than hand-written kernels; they are still not CoreMark."
    );
}
