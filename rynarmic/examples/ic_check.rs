//! A check that an exchange branch's inline cache really hits and jumps straight across.
//!
//! All three shapes are two instructions per round with no memory access:
//! A `b .`            a static self-loop (the direct link from `emit_link_site`, the baseline)
//! B `mov pc, lr`     an exchange branch to the block start (through the inline cache)
//! C `bx lr`          the same, differently encoded
use rynarmic::memory::SimpleMemory;
use rynarmic::{Cpsr, Cpu};
use std::time::Instant;

const CODE: u32 = 0x1000;

fn build(words: &[u32], lr: u32) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13);
    cpu.state.set_reg(14, lr);
    cpu.state.set_reg(0, 0);
    cpu
}

fn mips(words: &[u32], lr: u32, n: u64, jit: bool) -> f64 {
    let mut cpu = build(words, lr);
    if !jit {
        cpu.set_jit_enabled(false);
    }
    let t = Instant::now();
    cpu.run_cached(n).unwrap();
    n as f64 / t.elapsed().as_secs_f64() / 1e6
}

fn main() {
    let n = 20_000_000u64;
    for (name, words, lr) in [
        ("A b .        ", vec![0xE280_0001, 0xEAFF_FFFE], 0),
        ("B mov pc, lr ", vec![0xE280_0001, 0xE1A0_F00E], CODE),
        ("C bx lr      ", vec![0xE280_0001, 0xE12F_FF1E], CODE),
    ] {
        let j = mips(&words, lr, n, true);
        let i = mips(&words, lr, n, false);
        println!(
            "{name}: JIT {j:8.1} MIPS  (interpreted {i:6.1}, {:.1}x)",
            j / i
        );
    }
}
