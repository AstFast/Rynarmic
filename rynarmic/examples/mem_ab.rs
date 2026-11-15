//! An A/B of the access path: one piece of guest code and one memory implementation, switching the JIT's window inlining.
//!
//! Through the API (`set_fastmem_enabled`) rather than an environment variable — the latter does not exist,
//! so the earlier "callback" numbers were bogus (both runs had windows on).
use rynarmic::memory::SimpleMemory;
use rynarmic::{Cpsr, Cpu};
use std::time::Instant;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;

fn mips(words: &[u32], n: u64, fastmem: bool) -> f64 {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.cpsr = Cpsr::from_bits(0x13);
    cpu.state.set_reg(2, DATA);
    cpu.state.set_reg(13, 0x3000);
    cpu.state.set_reg(14, CODE);
    cpu.set_fastmem_enabled(fastmem);
    let t = Instant::now();
    cpu.run_cached(n).unwrap();
    n as f64 / t.elapsed().as_secs_f64() / 1e6
}

fn main() {
    let n = 20_000_000u64;
    let stack: Vec<u32> = vec![0xE92D_4030, 0xE280_0001, 0xE592_1000, 0xE8BD_8030];
    let data: Vec<u32> = vec![
        0xE592_1000,
        0xE281_1001,
        0xE582_1000,
        0xE280_0001,
        0xE350_0000,
        0x1AFF_FFFA,
    ];
    println!(
        "{:<22} {:>10} {:>10} {:>8}",
        "shape", "window", "callback", "delta"
    );
    for (name, words) in [
        ("stack (3 mem/round)", &stack),
        ("data (2 mem/round)", &data),
    ] {
        let w = mips(words, n, true);
        let c = mips(words, n, false);
        println!(
            "{name:<22} {w:>10.1} {c:>10.1} {:>7.0}%",
            (w - c) / c * 100.0
        );
    }
}
