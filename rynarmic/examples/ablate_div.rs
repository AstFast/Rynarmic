//! An ablation over the hot block of `kern_div`: change a few instructions' encodings and measure what each part really costs.
//!
//! The `__udivsi3` main loop (0x1168..0x11a4, sixteen instructions) has three parts:
//! 1. four `cmp; subcs; orrcs` groups (eight **conditionally executed** ALUs, which the backend commits with cmov);
//! 2. `cmp r0,#0` plus an `lsrsne` that **writes the flags conditionally** (the backend pays for a flag materialization there);
//! 3. the `bne` self-loop back edge.
//!
//! The ablation erases the condition bits (`cs` -> `al`, `ne` -> `al`). Loop termination is decided by a shift of `r3`,
//! so erasing the condition bits does not change the loop count's order of magnitude; but the data flow does change, so **the checksum changes** —
//! what is compared is time per instruction (MIPS), not absolute time. In the `lsrsne -> lsr` arm the loop
//! may run longer (the early exit is gone), so the reading means "time per instruction".
//!
//! Usage: `cargo run --release --example ablate_div`

include!(concat!(env!("OUT_DIR"), "/compiled_bench.rs"));

use std::time::{Duration, Instant};

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

const MEMORY_SIZE: usize = 0x40_0000;
const STACK: u32 = 0x3F_000;
const LIMIT: u64 = 40_000_000;
const ROUNDS: usize = 5;

/// `cs` (0x2) -> `al` (0xE): only the condition field changes, the rest of the encoding does not.
const fn al(addr: u32, word: u32) -> (u32, u32) {
    (addr, (word & 0x0FFF_FFFF) | 0xE000_0000)
}

/// The eight conditionally executed ALUs of the four `cmp; subcs; orrcs` groups (the unconditional `cmp`s do not count).
const COND_ALU: [(u32, u32); 8] = [
    al(0x116c, 0x2040_0001),
    al(0x1170, 0x2182_2003),
    al(0x1178, 0x2040_00a1),
    al(0x117c, 0x2182_20a3),
    al(0x1184, 0x2040_0121),
    al(0x1188, 0x2182_2123),
    al(0x1190, 0x2040_01a1),
    al(0x1194, 0x2182_21a3),
];

/// `lsrsne` (ne, writes the flags) -> `lsrs` (al, writes the flags): removes the guard and the materialization.
const UNCOND_FLAGS: (u32, u32) = (0x119c, 0xe1b0_3223);

fn new_cpu(patches: &[(u32, u32)]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, MEMORY_SIZE);
    memory.write_bytes(BASE, BLOB);
    for (addr, word) in patches {
        memory.write_u32(*addr, *word);
    }
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(ENTRY);
    cpu
}

fn enter(cpu: &mut Cpu<SimpleMemory>) {
    cpu.state.set_pc(ENTRY);
    cpu.state.set_reg(13, STACK);
    cpu.state.set_reg(0, 0); // which = kern_div
    cpu.state.set_reg(1, 1000);
    cpu.state.set_reg(2, 0xDEAD_BEEF);
}

/// Runs ROUNDS rounds on one Cpu and takes the best (blocks stay compiled across rounds, so translation cost is not folded in).
fn measure(name: &str, patches: &[(u32, u32)]) -> (f64, u64, u32) {
    let mut cpu = new_cpu(patches);
    // One warm-up round, to get the blocks compiled
    enter(&mut cpu);
    let _ = cpu.run_cached_counted(LIMIT);

    let mut best = Duration::MAX;
    let mut count = 0;
    let mut checksum = 0;
    for _ in 0..ROUNDS {
        enter(&mut cpu);
        let start = Instant::now();
        let (executed, result) = cpu.run_cached_counted(LIMIT);
        best = best.min(start.elapsed());
        assert!(result.is_err(), "the kernel should end with a swi");
        count = executed;
        checksum = cpu.state.reg(0);
    }
    let mips = count as f64 / best.as_secs_f64() / 1_000_000.0;
    println!("{name:<30} {count:>8} insns  {mips:7.1} MIPS  checksum {checksum:#010x}");
    (mips, count, checksum)
}

fn main() {
    if !AVAILABLE {
        println!("no ARM cross toolchain, skipping.");
        return;
    }
    println!(
        "best of {ROUNDS} rounds per arm; the ablated arms change the data flow, so a different checksum is expected.\n"
    );

    let (base, _, _) = measure("baseline (the real code)", &[]);
    let (cond, _, _) = measure("8 conditional ALUs -> unconditional", &COND_ALU);
    let (tail, _, _) = measure(
        "lsrsne -> lsrs (no guard, no materialization)",
        &[UNCOND_FLAGS],
    );
    let mut all = COND_ALU.to_vec();
    all.push(UNCOND_FLAGS);
    let (both, _, _) = measure("both removed", &all);

    println!(
        "\n== readings (compared per instruction; MIPS already divided by each arm's instruction count) =="
    );
    println!(
        "conditional ALUs (cmov commit): {:+6.1}%   {base:.1} -> {cond:.1}",
        100.0 * (cond - base) / base
    );
    println!(
        "tail guard + materialization:  {:+6.1}%   {base:.1} -> {tail:.1}",
        100.0 * (tail - base) / base
    );
    println!(
        "both together:                 {:+6.1}%   {base:.1} -> {both:.1}",
        100.0 * (both - base) / base
    );
}
