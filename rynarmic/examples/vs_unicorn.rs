//! Throughput against unicorn (QEMU TCG) on the same workload: one piece of ARM machine code, one initial state.
//!
//! Usage: `cargo run --release --example vs_unicorn`
//!
//! Two synthetic workloads plus one real one (`__aeabi_uidiv` compiled with `arm-none-eabi-gcc -O2`).
//! The real workload's image is also written to `target/uc_triage/kernel.bin`, so that a dynarmic comparison program
//! can read the same bytes.
//!
//! Method: best of `ROUNDS` rounds per arm; unicorn stops at the loop exit with `until`, avoiding count's
//! per-instruction counting overhead. Machine noise is substantial (20% between runs of one binary), so ratios are more trustworthy than absolutes.

use std::time::Instant;

use rynarmic::{Cpu, memory::SimpleMemory};

include!(concat!(env!("OUT_DIR"), "/compiled_bench.rs"));

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
const ROUNDS: usize = 5;

/// The synthetic workloads' iteration count: tens of millions of instructions per arm, making translation cost negligible.
const WORK: u32 = 8_000_000;

/// The real kernel's (`__aeabi_uidiv`) executed count and initial state, as in compiled_bench.
const KERNEL_STACK: u32 = 0x3F_000;
const KERNEL_MEMORY: usize = 0x100_0000;
const KERNEL_INSTRUCTIONS: u64 = 165_919;

// ---------------------------------------------------------------------------
// The unicorn FFI (only the smallest subset is used)
// ---------------------------------------------------------------------------

#[cfg(unicorn_ref)]
mod ffi {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};

    pub const UC_ARCH_ARM: u32 = 1;
    pub const UC_MODE_ARM: u32 = 0;
    pub const UC_PROT_ALL: u32 = 7;
    pub const UC_ARM_REG_CPSR: c_int = 3;
    pub const UC_ARM_REG_LR: c_int = 10;
    pub const UC_ARM_REG_PC: c_int = 11;
    pub const UC_ARM_REG_SP: c_int = 12;
    pub fn reg(n: u32) -> c_int {
        66 + n as c_int
    }

    unsafe extern "C" {
        pub fn uc_open(arch: u32, mode: u32, engine: &mut *mut c_void) -> c_int;
        pub fn uc_close(engine: *mut c_void) -> c_int;
        pub fn uc_mem_map(engine: *mut c_void, address: u64, size: usize, perms: u32) -> c_int;
        pub fn uc_mem_write(
            engine: *mut c_void,
            address: u64,
            bytes: *const u8,
            size: usize,
        ) -> c_int;
        pub fn uc_reg_write(engine: *mut c_void, regid: c_int, value: *const c_void) -> c_int;
        pub fn uc_emu_start(
            engine: *mut c_void,
            begin: u64,
            until: u64,
            timeout: u64,
            count: usize,
        ) -> c_int;
        fn uc_strerror(code: c_int) -> *const c_char;
    }

    pub fn err_text(code: c_int) -> String {
        unsafe {
            CStr::from_ptr(uc_strerror(code))
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// One workload: the image to write into guest memory, the initial registers, and the stop address.
struct Workload {
    name: &'static str,
    /// `(address, bytes)`, written in order
    image: Vec<(u32, Vec<u8>)>,
    memory_size: usize,
    /// `(register number, value)` for r0..r12
    init: Vec<(u32, u32)>,
    pc: u32,
    /// How many guest instructions to execute
    instructions: u64,
    /// unicorn's stop address (the loop exit), which is also the PC when our budget runs out
    #[cfg_attr(not(unicorn_ref), allow(dead_code))]
    until: u64,
}

fn words(at: u32, list: &[u32]) -> (u32, Vec<u8>) {
    (at, list.iter().flat_map(|w| w.to_le_bytes()).collect())
}

fn workloads() -> Vec<Workload> {
    let mut list = Vec::new();

    // (1) A pure-arithmetic tight loop: subs, bne, two instructions per iteration.
    {
        let code = [0xE250_0001u32, 0x1AFF_FFFD, 0xEAFF_FFFE];
        list.push(Workload {
            name: "arithmetic tight loop (2 per iteration)",
            image: vec![words(CODE, &code)],
            memory_size: 0x20_000,
            init: vec![(0, WORK)],
            pc: CODE,
            instructions: WORK as u64 * 2,
            until: (CODE + 8) as u64,
        });
    }

    // (2) An access loop: ldr, and, subs, bne, four per iteration, with the pointer wrapping inside a 16 KiB window.
    {
        let code = [
            0xE491_0004u32, // ldr r0, [r1], #4
            0xE001_1003,    // and r1, r1, r3
            0xE252_2001,    // subs r2, r2, #1
            0x1AFF_FFFB,    // bne 0x1000
            0xEAFF_FFFE,    // b .
        ];
        let pattern: Vec<u32> = (0..1024u32).map(|i| 0x1111_0000 + i).collect();
        list.push(Workload {
            name: "access loop (4 per iteration)",
            image: vec![words(CODE, &code), words(DATA, &pattern)],
            memory_size: 0x20_000,
            init: vec![(1, DATA), (2, WORK), (3, 0x3FFF)],
            pc: CODE,
            instructions: WORK as u64 * 4,
            until: (CODE + 16) as u64,
        });
    }

    // (3) A real kernel: __aeabi_uidiv at -O2 (1000 / 0xDEADBEEF).
    if AVAILABLE {
        // A bare-metal kernel ends with `swi 0` (compiled_bench asserts "should end with a swi" too):
        // find the last swi and use it as unicorn's stop address.
        let exit = BLOB
            .windows(4)
            .rposition(|w| w == [0x00, 0x00, 0x00, 0xEF])
            .map(|off| BASE as u64 + off as u64)
            .unwrap_or(BASE as u64);
        list.push(Workload {
            name: "real kernel uidiv",
            image: vec![(BASE, BLOB.to_vec())],
            memory_size: KERNEL_MEMORY,
            init: vec![
                (0, 0),
                (1, 1000),
                (2, 0xDEAD_BEEF),
                (3, 0),
                (13, KERNEL_STACK),
            ],
            pc: ENTRY,
            instructions: KERNEL_INSTRUCTIONS,
            until: exit,
        });
    }

    list
}

// ---------------------------------------------------------------------------
// The three arms
// ---------------------------------------------------------------------------

fn rynarmic_cpu(work: &Workload) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, work.memory_size);
    for (addr, bytes) in &work.image {
        memory.write_bytes(*addr, bytes);
    }
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(work.pc);
    for &(r, v) in &work.init {
        cpu.state.set_reg(r as u8, v);
    }
    cpu
}

fn run_once(cpu: &mut Cpu<SimpleMemory>, work: &Workload, cached: bool) {
    for (addr, bytes) in &work.image {
        cpu.memory.write_bytes(*addr, bytes);
    }
    cpu.state.set_pc(work.pc);
    for &(r, v) in &work.init {
        cpu.state.set_reg(r as u8, v);
    }
    // The real kernel ends with `swi 0`, so an exception means "it finished" and either outcome is accepted.
    if cached {
        let _ = cpu.run_cached_counted(work.instructions);
    } else {
        let _ = cpu.run(work.instructions);
    }
}

fn bench_rynarmic(work: &Workload, cached: bool) -> f64 {
    let mut cpu = rynarmic_cpu(work);
    // Warm up: get the blocks compiled, so translation cost stays out of throughput
    run_once(&mut cpu, work, cached);

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        for (addr, bytes) in &work.image {
            cpu.memory.write_bytes(*addr, bytes);
        }
        cpu.state.set_pc(work.pc);
        for &(r, v) in &work.init {
            cpu.state.set_reg(r as u8, v);
        }
        let start = Instant::now();
        if cached {
            let _ = cpu.run_cached_counted(work.instructions);
        } else {
            let _ = cpu.run(work.instructions);
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    work.instructions as f64 / best / 1e6
}

#[cfg(unicorn_ref)]
fn bench_unicorn(work: &Workload) -> Result<f64, String> {
    use ffi::*;
    unsafe {
        let mut engine: *mut std::os::raw::c_void = std::ptr::null_mut();
        let rc = uc_open(UC_ARCH_ARM, UC_MODE_ARM, &mut engine);
        if rc != 0 {
            return Err(err_text(rc));
        }
        let rc = uc_mem_map(engine, 0, work.memory_size, UC_PROT_ALL);
        if rc != 0 {
            return Err(err_text(rc));
        }
        for (addr, bytes) in &work.image {
            uc_mem_write(engine, *addr as u64, bytes.as_ptr(), bytes.len());
        }

        let mut best = f64::INFINITY;
        // One warm-up round (so unicorn translates its TBs), untimed
        for round in 0..=ROUNDS {
            let pc = work.pc;
            uc_reg_write(engine, UC_ARM_REG_PC, &pc as *const u32 as *const _);
            let cpsr: u32 = 0x13;
            uc_reg_write(engine, UC_ARM_REG_CPSR, &cpsr as *const u32 as *const _);
            // Guest register number to unicorn number: r0-r12 = 66+n, SP = 12, LR = 10, PC = 11
            for n in 0..15u32 {
                let id = match n {
                    13 => UC_ARM_REG_SP,
                    14 => UC_ARM_REG_LR,
                    _ => reg(n),
                };
                let v = work
                    .init
                    .iter()
                    .find(|(r, _)| *r == n)
                    .map(|(_, v)| *v)
                    .unwrap_or(0);
                uc_reg_write(engine, id, &v as *const u32 as *const _);
            }
            let start = Instant::now();
            let rc = uc_emu_start(engine, work.pc as u64, work.until, 0, 0);
            let secs = start.elapsed().as_secs_f64();
            if rc != 0 {
                return Err(format!("uc_emu_start: {}", err_text(rc)));
            }
            if round > 0 {
                best = best.min(secs);
            }
        }
        uc_close(engine);
        Ok(work.instructions as f64 / best / 1e6)
    }
}

#[cfg(not(unicorn_ref))]
fn bench_unicorn(_work: &Workload) -> Result<f64, String> {
    Err("no unicorn (set RYNARMIC_UNICORN to an unpacked unicorn package)".into())
}

fn main() {
    // Leave an identical image for a dynarmic comparison program
    if AVAILABLE {
        let dir = std::path::Path::new("target/uc_triage");
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join("kernel.bin"), BLOB);
    }

    println!(
        "one piece of guest machine code, one initial state; best of {ROUNDS} rounds per arm; MIPS.\n"
    );
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "workload", "interpreter", "JIT", "unicorn", "JIT/unicorn", "JIT speedup"
    );
    println!("{}", "-".repeat(80));

    for work in workloads() {
        let interp = bench_rynarmic(&work, false);
        let jit = bench_rynarmic(&work, true);
        let uc = bench_unicorn(&work);
        let (uc_text, ratio) = match &uc {
            Ok(mips) => (format!("{mips:>10.1}"), format!("{:>11.2}×", jit / mips)),
            Err(_) => (format!("{:>10}", "—"), format!("{:>12}", "—")),
        };
        println!(
            "{:<22} {:>10.1} {:>10.1} {uc_text} {ratio} {:>9.1}×",
            work.name,
            interp,
            jit,
            jit / interp
        );
        if let Err(e) = uc {
            println!("   unicorn did not run: {e}");
        }
    }
}
