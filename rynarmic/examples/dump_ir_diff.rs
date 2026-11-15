//! Triage aid: find the first diverging instruction in a failing differential case.
//!
//! Usage: `cargo run --release --example dump_ir_diff`
#![allow(dead_code, unused_variables, unused_mut, clippy::collapsible_if)]

use rynarmic::{Cpsr, Cpu, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const DATA_START: u32 = 0x2000;
const MEMORY_SIZE: usize = 0x8000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn new_cpu(program: &[u32], rng: &mut Rng) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, MEMORY_SIZE);
    memory.write_words(CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    for reg in 0..13u8 {
        let value = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
        cpu.state.set_reg(reg, value);
    }
    let flags = rng.next() as u32 & 0xF000_0000;
    cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() | flags);
    cpu
}

fn snapshot(cpu: &Cpu<SimpleMemory>) -> String {
    let regs: Vec<String> = cpu.state.r.iter().map(|v| format!("{v:08x}")).collect();
    format!("{} cpsr={:08x}", regs.join(","), cpu.state.cpsr.bits())
}

fn main() {
    let program: Vec<u32> = vec![
        0xE58C5078, 0xE050B007, 0x558480A4, 0xE59670D9, 0xE586104C, 0xE01B5005, 0xE0040990,
        0xE1B03026, 0xE0129005, 0xE0D26008, 0x1A000023, 0xE1B06040, 0xE1B05025, 0xE5D590E1,
        0xE3A0A0C3, 0x0A00002C, 0xD58C1013, 0xE0508007, 0xEAFFFFF9, 0xE1B04614, 0xE023400B,
        0xE59F4019, 0xE5841088, 0xE1B03040, 0x1A000006, 0xE28F0011, 0xE1B09417, 0xE0575009,
        0xE0507003, 0xE021C005, 0xE284A075, 0x7583C053, 0xE0D8B006, 0xE3A030E6, 0xE0D4800C,
        0xE1E09007, 0x0AFFFFFF, 0xE1B0204B, 0x45887091, 0xE35000EF, 0xE0060895, 0xE2869052,
        0xE28C0074, 0xE28A3059, 0x1AFFFFDF, 0x0A000046, 0xE057B002, 0xE023500A, 0xE1B05000,
        0xE00C9002, 0xE0077001, 0xE0151003, 0xE00CA00B, 0xE005B006, 0xE5DCA038, 0xE0050590,
        0xE358001F, 0xE0060294, 0xEAFFFFD0, 0xE580108A, 0xE0D5A005, 0xE58490B5, 0xE0B4900A,
        0xE28F80BA, 0xE1B0B57C, 0xE35600B2, 0xE3A070DF, 0xE0DB700C, 0xE05C100B, 0x31B0B003,
        0xE027A004, 0xE1B06858, 0xE0DC500B, 0x058A3036, 0xE1B02437, 0x728B304D, 0xE59F5050,
        0xE5D49031, 0xE1B07535, 0xE0D87008, 0xE02AC009, 0xE02A4009, 0xE59F30F9, 0xE1E03000,
        0xE59A40D5, 0xE1B0B048, 0xE1B06770, 0xE1B02005, 0xE5932027, 0xE1B04C56, 0xE012C004,
        0xE1B08134, 0xB5834084, 0xE5D010D2, 0xE0DBB009, 0xE594B0EF, 0xE3A070B4, 0xE0010599,
        0xC1B06004, 0x8285704E, 0xA2806028, 0xC589300E, 0x658C80B7, 0xE05B1005, 0xE1B0C006,
        0xE586C0AE, 0xE1B03040, 0xE3A0800F, 0xB58C9031, 0xA581A0EF, 0xE000300C, 0x71B05002,
        0xE584C0BE, 0x828A0074, 0xE0130005, 0xE0D27004, 0xE3A00027, 0xE5875070, 0xE584A0B6,
        0xE3560020,
    ];

    // The same seeding protocol as the differential test: the init of case 1
    // Case 1's actual init, as recorded from the failure message
    {
        let init: u64 = 0xd2b70134096af2d2;
        let mut cached = new_cpu(&program, &mut Rng(init));
        let mut stepped = new_cpu(&program, &mut Rng(init));

        // The pc and r11 trace of the first 59 single-step steps
        {
            let mut tr = new_cpu(&program, &mut Rng(init));
            for i in 0..59u64 {
                let pc = tr.state.pc();
                let idx = ((pc.wrapping_sub(CODE)) / 4) as usize;
                if let Some(&w) = program.get(idx) {
                    let writes_r11 = (w >> 12) & 0xF == 11 || (w >> 16) & 0xF == 11;
                    if writes_r11 || i >= 55 {
                        println!(
                            "step {i}: pc={pc:#x} word={w:08X} r11={:08x}",
                            tr.state.r[11]
                        );
                    }
                }
                tr.run(1).unwrap();
            }
        }
        let _ = &stepped;
        // Comparing point by point on executed count: run_cached_counted(k) may actually execute e_k instructions because of block
        // granularity overshooting k, so the single-step path is topped up to the same e_k before comparing. Take the earliest divergence.
        let mut first = None;
        for k in [28u64] {
            let mut c = new_cpu(&program, &mut Rng(init));
            let (executed, _) = c.run_cached_counted(k);
            let mut st = new_cpu(&program, &mut Rng(init));
            st.run(executed).unwrap();
            if snapshot(&c) != snapshot(&st) {
                first = Some((k, executed));
                break;
            }
        }
        match first {
            Some((k, executed)) => {
                let mut c = new_cpu(&program, &mut Rng(init));
                let _ = c.run_cached_counted(k);
                let mut st = new_cpu(&program, &mut Rng(init));
                let _ = st.run(executed);
                // Memory differences
                let mut diffs = Vec::new();
                let mut addr = DATA_START;
                while addr < MEMORY_SIZE as u32 {
                    let a = c.memory.peek_u32(addr);
                    let b = st.memory.peek_u32(addr);
                    if a != b {
                        diffs.push(format!("{addr:#x}: block {a:08x} vs step {b:08x}"));
                    }
                    addr += 4;
                }
                println!("{} memory differences: {}", diffs.len(), diffs.join("; "));
                // Find the block that produced the tail at 0x1174 and print its IR (with the r11 write highlighted)
                let tail = c.state.pc();
                let mut start = CODE;
                let mut found = None;
                while start < tail {
                    if let Ok(rynarmic::block::Translation::Block(b)) = rynarmic::block::translate(
                        &mut new_cpu(&program, &mut Rng(1)).memory,
                        start,
                        false,
                    ) {
                        if b.next_pc == tail || (b.start_pc <= tail && b.next_pc > tail) {
                            found = Some(b);
                        }
                    }
                    start += 4;
                }
                if let Some(b) = found {
                    println!(
                        "source block pc={:#x} instructions={} next={:#x}",
                        b.start_pc, b.instruction_count, b.next_pc
                    );
                    for (j, op) in b.ops.iter().enumerate() {
                        let mark = if format!("{op:?}").contains("Some(11)")
                            || format!("{op:?}").contains("a: 11")
                            || format!("{op:?}").contains("b: 11")
                        {
                            " <<<< r11"
                        } else {
                            ""
                        };
                        println!("  [{j}] {op:?}{mark}");
                    }
                }
                let pc = c.state.pc();
                let index = ((pc.wrapping_sub(CODE)) / 4) as usize;
                println!(
                    "earliest divergence: budget {k}, {executed} executed, block tail pc={pc:#x}"
                );
                println!(
                    "  tail instruction word={:08X?}",
                    program.get(index.saturating_sub(1))
                );
                println!("  block: {}", snapshot(&c));
                println!("  step : {}", snapshot(&st));
            }
            None => println!("no divergence within 128 instructions (could not reproduce)"),
        }
    }
}
