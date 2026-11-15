//! Correctness checks for the realistic-shape kernels.
//!
//! These three are hand-assembled (there is no ARM cross toolchain; see the note in `support/kernels.rs`),
//! so **expected values** back them up: a wrong answer means the encodings are wrong, rather than "both paths are wrong together".
//!
//! Each kernel runs both paths (interpreter and block cache) and the blocks have to be really compiled.

#[path = "support/kernels.rs"]
mod kernels;

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

fn new_cpu(program: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(kernels::CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(kernels::CODE);
    cpu
}

/// Runs one kernel to `instructions` on both paths, returning `(registers, cpsr, compiled blocks)`.
fn run_both(
    program: &[u32],
    instructions: u64,
    setup: impl Fn(&mut Cpu<SimpleMemory>),
) -> Vec<([u32; 16], u32, usize)> {
    let mut out = Vec::new();
    for jit in [false, true] {
        let mut cpu = new_cpu(program);
        setup(&mut cpu);
        cpu.set_jit_enabled(jit);
        if jit {
            let mut executed = 0;
            while executed < instructions {
                executed += cpu.run_cached(instructions - executed).unwrap();
            }
        } else {
            cpu.run(instructions).unwrap();
        }
        out.push((
            cpu.state.r,
            cpu.state.cpsr.bits(),
            cpu.compiled_block_count(),
        ));
    }
    out
}

/// The comparable visible state, with **r15 excluded**.
///
/// `run_cached` works in blocks, and overshooting `run` by a few instructions or a whole block is the documented semantics,
/// so the two paths may stop at different PCs; r0-r14 and CPSR must match bit for bit.
fn comparable(run: &([u32; 16], u32, usize)) -> ([u32; 15], u32) {
    let mut regs = [0u32; 15];
    regs.copy_from_slice(&run.0[..15]);
    (regs, run.1)
}

#[test]
fn udiv_kernel_divides_correctly() {
    let program = kernels::udiv_program();

    for (dividend, divisor) in [
        (0xFFFF_FFFFu32, 7u32),
        (1_000_000, 3),
        (0, 5),
        (1, 1),
        (0x8000_0000, 2),
        (12345, 12345),
    ] {
        let runs = run_both(&program, kernels::UDIV_INSTRUCTIONS, |cpu| {
            cpu.state.set_reg(0, dividend);
            cpu.state.set_reg(1, divisor);
        });

        for (index, (regs, _, blocks)) in runs.iter().enumerate() {
            let path = if index == 0 { "interpreter" } else { "JIT" };
            assert_eq!(
                (regs[0], regs[2]),
                (dividend / divisor, dividend % divisor),
                "{path}: {dividend} / {divisor}"
            );
            if index == 1 {
                assert!(
                    *blocks > 0,
                    "the division kernel should compile to machine code"
                );
            }
        }
        // The two paths must agree bit for bit, flags included
        assert_eq!(
            comparable(&runs[0]),
            comparable(&runs[1]),
            "{dividend} / {divisor}: the two paths disagree"
        );
    }
}

#[test]
fn memcpy_kernel_copies_words() {
    let program = kernels::memcpy_program();
    let words = 64u32;

    let runs = run_both(&program, (words / 4) as u64 * 10, |cpu| {
        cpu.state.set_reg(0, kernels::DST);
        cpu.state.set_reg(1, kernels::SRC);
        cpu.state.set_reg(2, words);
        for index in 0..words {
            cpu.memory
                .write_u32(kernels::SRC + index * 4, 0x1000_0000 + index);
        }
    });

    // run_both returns registers only, so memory is checked separately by rerunning the JIT path
    let mut cpu = new_cpu(&program);
    cpu.state.set_reg(0, kernels::DST);
    cpu.state.set_reg(1, kernels::SRC);
    cpu.state.set_reg(2, words);
    for index in 0..words {
        cpu.memory
            .write_u32(kernels::SRC + index * 4, 0x1000_0000 + index);
    }
    cpu.run_cached((words / 4) as u64 * 10).unwrap();
    assert!(
        cpu.compiled_block_count() > 0,
        "the copy kernel should be compiled"
    );

    for index in 0..words {
        assert_eq!(
            cpu.memory.peek_u32(kernels::DST + index * 4),
            0x1000_0000 + index,
            "word {index} was not copied correctly"
        );
    }
    assert_eq!(
        cpu.state.reg(1),
        kernels::SRC + words * 4,
        "the source pointer should advance to the end"
    );
    assert_eq!(
        cpu.state.reg(0),
        kernels::DST + words * 4,
        "the destination pointer should advance to the end"
    );
    assert_eq!(
        comparable(&runs[0]),
        comparable(&runs[1]),
        "the two paths disagree"
    );
}

#[test]
fn call_kernel_sums_through_bl_and_bx() {
    let program = kernels::call_program_with_function();

    let runs = run_both(&program, kernels::CALL_INSTRUCTIONS, |_| {});

    // Σ(k+3), k = 1..=200
    let expected: u32 = (1..=200u32).map(|k| k + 3).sum();
    for (index, (regs, _, blocks)) in runs.iter().enumerate() {
        let path = if index == 0 { "interpreter" } else { "JIT" };
        assert_eq!(regs[5], expected, "{path}: the accumulated result");
        assert_eq!(regs[4], 0, "{path}: the counter is back to zero");
        if index == 1 {
            assert!(*blocks > 0, "the call kernel should be compiled");
        }
    }
    assert_eq!(
        comparable(&runs[0]),
        comparable(&runs[1]),
        "the two paths disagree"
    );
}
