//! Genuinely compiled ARM kernels (`arm-none-eabi-gcc`; see `build.rs` and `support/prog/`).
//!
//! Unlike `real_workload.rs`, whose kernels are hand-assembled, this is real machine code produced from C:
//! it goes through `__aeabi_uidiv` and exhibits whatever block shapes and register allocation the compiler chose.
//!
//! Every expected value is recomputed in Rust rather than relying on "the two paths agree", so a wrong compiler optimization or a
//! wrong emulator both surface here. Without a toolchain the cases return early and say why.

include!(concat!(env!("OUT_DIR"), "/compiled_bench.rs"));

use rynarmic::{Cpu, Memory, cpu::ExceptionKind, memory::SimpleMemory};

const MEMORY_SIZE: usize = 0x40_0000;
/// The stack sits at the top of memory.
const STACK: u32 = 0x3F_000;
/// The copy kernel's destination; its source sits at 0x20000, with 16 KiB between them.
const DST: u32 = 0x4_000;
const SRC: u32 = 0x2_0000;
/// A per-run instruction ceiling, so a miswritten case cannot hang the suite.
const LIMIT: u64 = 40_000_000;

fn expect_div(rounds: u32, seed: u32) -> u32 {
    let mut acc = seed;
    for i in 1..=rounds {
        acc = acc.wrapping_add(0xFFFF_0000u32 / (i | 1));
        acc ^= acc >> 3;
        if acc & 1 != 0 {
            acc = acc.wrapping_add(0x9E37_79B9);
        } else {
            acc = acc.wrapping_sub(0x7F4A_7C15);
        }
    }
    acc
}

fn expect_copy(src: &[u32]) -> u32 {
    let mut sum = 0u32;
    for (i, value) in src.iter().enumerate() {
        sum = sum.wrapping_add(value.wrapping_add(i as u32));
    }
    sum
}

fn expect_call(rounds: u32, seed: u32) -> u32 {
    let mut acc = seed;
    for i in 0..rounds {
        let x = acc.wrapping_add(i);
        acc = (x << 1) ^ (x >> 3) ^ 0x1234;
    }
    acc
}

/// Runs one kernel and returns `(return value, executed count, cpu)`.
fn run(
    which: u32,
    args: [u32; 3],
    jit: bool,
    prep: impl Fn(&mut Cpu<SimpleMemory>),
) -> (u32, u64, Cpu<SimpleMemory>) {
    let mut memory = SimpleMemory::new(0, MEMORY_SIZE);
    memory.write_bytes(BASE, BLOB);
    let mut cpu = Cpu::new(memory);
    cpu.set_jit_enabled(jit);

    cpu.state.set_pc(ENTRY);
    cpu.state.set_reg(13, STACK);
    cpu.state.set_reg(0, which);
    cpu.state.set_reg(1, args[0]);
    cpu.state.set_reg(2, args[1]);
    cpu.state.set_reg(3, args[2]);
    prep(&mut cpu);

    let (executed, result) = cpu.run_cached_counted(LIMIT);
    match result {
        // start.s ends with `swi #0`, which is the "it finished" signal
        Err(ExceptionKind::SoftwareInterrupt(0)) => {}
        Err(other) => {
            panic!("the kernel ended abnormally: {other:?} (after {executed} instructions)")
        }
        Ok(()) => panic!("the kernel hit the {LIMIT}-instruction ceiling without finishing"),
    }
    if jit {
        assert!(
            cpu.compiled_block_count() > 0,
            "the compiled code should be taken up by the JIT"
        );
    }
    (cpu.state.reg(0), executed, cpu)
}

#[test]
fn compiled_div_kernel_matches_rust() {
    if !AVAILABLE {
        eprintln!(
            "skipped: no ARM cross toolchain (set RYNARMIC_ARM_GCC, or put arm-none-eabi-gcc on PATH)"
        );
        return;
    }
    for (rounds, seed) in [(1u32, 0u32), (64, 12345), (1000, 0xDEAD_BEEF)] {
        let (result, executed, _) = run(0, [rounds, seed, 0], true, |_| {});
        assert_eq!(
            result,
            expect_div(rounds, seed),
            "rounds={rounds} seed={seed:#X}"
        );
        assert!(
            executed > rounds as u64,
            "at least one instruction per iteration"
        );

        let (reference, _, _) = run(0, [rounds, seed, 0], false, |_| {});
        assert_eq!(result, reference, "the JIT and the interpreter disagree");
    }
}

#[test]
fn compiled_copy_kernel_matches_rust() {
    if !AVAILABLE {
        eprintln!("skipped: no ARM cross toolchain");
        return;
    }
    const WORDS: u32 = 256;
    let src: Vec<u32> = (0..WORDS).map(|i| i.wrapping_mul(2_654_435_761)).collect();

    let (result, _, cpu) = run(1, [DST, SRC, WORDS], true, |cpu| {
        for (i, value) in src.iter().enumerate() {
            cpu.memory.write_u32(SRC + i as u32 * 4, *value);
        }
    });

    assert_eq!(result, expect_copy(&src), "the copy kernel's checksum");
    for (i, value) in src.iter().enumerate() {
        assert_eq!(
            cpu.memory.peek_u32(DST + i as u32 * 4),
            value.wrapping_add(i as u32),
            "word {i} was not copied correctly"
        );
    }

    let (reference, _, _) = run(1, [DST, SRC, WORDS], false, |cpu| {
        for (i, value) in src.iter().enumerate() {
            cpu.memory.write_u32(SRC + i as u32 * 4, *value);
        }
    });
    assert_eq!(result, reference, "the JIT and the interpreter disagree");
}

#[test]
fn compiled_call_kernel_matches_rust() {
    if !AVAILABLE {
        eprintln!("skipped: no ARM cross toolchain");
        return;
    }
    for (rounds, seed) in [(1u32, 7u32), (2000, 0x1234_5678)] {
        let (result, _, _) = run(2, [rounds, seed, 0], true, |_| {});
        assert_eq!(result, expect_call(rounds, seed), "rounds={rounds}");

        let (reference, _, _) = run(2, [rounds, seed, 0], false, |_| {});
        assert_eq!(result, reference, "the JIT and the interpreter disagree");
    }
}
