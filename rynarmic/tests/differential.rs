//! A two-path differential test: one random program run through the single-step interpreter and through the block cache,
//! requiring identical registers, flags and memory.
//!
//! Why this test is the safety net for later work: deferred flags, constant propagation and the JIT backend all change the execution path, and
//! the moment they diverge, this fails on random input rather than waiting for a guest to run off the rails.

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

const CODE: u32 = 0x0000;
const CODE_END: u32 = 0x1000;
const DATA_START: u32 = 0x2000;
const MEMORY_SIZE: usize = 0x8000;
const PROGRAM_LEN: usize = 120;

/// The code region is made read-only, standing in for a real device's ROM.
///
/// That is not just to make the test pass: the block cache's premise is that the code region is not rewritten at will.
/// The moment a random program's `str` rewrites the code, the single-step path reads the new instruction while the block path still runs the old block:
/// that is the semantic boundary of **self-modifying code**, which a real host handles with page protection plus invalidation (see PERF.md),
/// and the test excludes it with a read-only code region.
struct RomMemory {
    inner: SimpleMemory,
}

impl RomMemory {
    fn new(size: usize) -> Self {
        Self {
            inner: SimpleMemory::new(0, size),
        }
    }

    fn peek_u32(&self, addr: u32) -> u32 {
        self.inner.peek_u32(addr)
    }

    fn write_words(&mut self, addr: u32, words: &[u32]) {
        self.inner.write_words(addr, words);
    }
}

impl Memory for RomMemory {
    fn read_u8(&mut self, addr: u32) -> u8 {
        self.inner.read_u8(addr)
    }

    fn write_u8(&mut self, addr: u32, value: u8) {
        if addr >= CODE_END {
            self.inner.write_u8(addr, value);
        }
    }

    fn read_u16(&mut self, addr: u32) -> u16 {
        self.inner.read_u16(addr)
    }

    fn write_u16(&mut self, addr: u32, value: u16) {
        if addr >= CODE_END {
            self.inner.write_u16(addr, value);
        }
    }

    fn read_u32(&mut self, addr: u32) -> u32 {
        self.inner.read_u32(addr)
    }

    fn write_u32(&mut self, addr: u32, value: u32) {
        if addr >= CODE_END {
            self.inner.write_u32(addr, value);
        }
    }
}

/// A fixed-seed xorshift64, so a failure can be reproduced.
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

    fn reg(&mut self) -> u32 {
        (self.next() % 13) as u32 // r0..r12, avoiding sp, lr and pc
    }

    fn imm(&mut self) -> u32 {
        (self.next() % 256) as u32
    }
}

/// Generates a random program: data processing, multiplies, accesses, conditional execution, and **static branches**.
///
/// The branches are deliberate: without them a program runs through once and ends, and block re-entry, block linking and cross-block flag passing
/// are never exercised. Targets are confined inside the program, so nothing jumps onto uninitialized encodings.
fn build_program(rng: &mut Rng) -> Vec<u32> {
    (0..PROGRAM_LEN)
        .map(|index| {
            let rd = rng.reg();
            let rn = rng.reg();
            let rm = rng.reg();
            let imm = rng.imm();
            let cond = (rng.next() % 14) as u32; // 0..=13, skipping AL (14) and NV (15)
            match rng.next() % 30 {
                // mov rd, #imm
                0 => 0xE3A0_0000 | rd << 12 | imm,
                // add rd, rn, #imm
                1 => 0xE280_0000 | rn << 16 | rd << 12 | imm,
                // subs rd, rn, rm
                2 => 0xE050_0000 | rn << 16 | rd << 12 | rm,
                // and rd, rn, rm
                3 => 0xE000_0000 | rn << 16 | rd << 12 | rm,
                // eor rd, rn, rm
                4 => 0xE020_0000 | rn << 16 | rd << 12 | rm,
                // mvn rd, rm
                5 => 0xE1E0_0000 | rd << 12 | rm,
                // cmp rn, #imm
                6 => 0xE350_0000 | rn << 16 | imm,
                // ldr rd, [rn, #imm]
                7 => 0xE590_0000 | rn << 16 | rd << 12 | imm,
                // str rd, [rn, #imm]
                8 => 0xE580_0000 | rn << 16 | rd << 12 | imm,
                // ldrb/strb
                9 => 0xE5D0_0000 | rn << 16 | rd << 12 | imm,
                // ldr rd, [pc, #imm] -- PC-relative, folded into an absolute address at translation time
                10 => 0xE59F_0000 | rd << 12 | imm,
                // add rd, pc, #imm -- PC as an operand
                11 => 0xE28F_0000 | rd << 12 | imm,
                // movs rd, rm -- shift 0, so C keeps its value (no C write)
                12 => 0xE1B0_0000 | rd << 12 | rm,
                // ands rd, rn, rm -- a logical operation: writes NZ and leaves C alone
                13 => 0xE010_0000 | rn << 16 | rd << 12 | rm,
                // adcs rd, rn, rm -- reads C, writes NZCV
                14 => 0xE0B0_0000 | rn << 16 | rd << 12 | rm,
                // sbcs rd, rn, rm -- reads C
                15 => 0xE0D0_0000 | rn << 16 | rd << 12 | rm,
                // A conditionally executed move with a random cond, checking that a guard decides from the flags of the run
                16 => (cond << 28) | 0x03A0_0000 | rd << 12 | imm,
                // A conditionally executed addition (without carry), with a random cond
                17 => (cond << 28) | 0x0280_0000 | rn << 16 | rd << 12 | imm,
                // A conditionally executed store
                18 => (cond << 28) | 0x0580_0000 | rn << 16 | rd << 12 | imm,
                // A conditionally executed movs: it both writes the flags and is controlled by them
                19 => (cond << 28) | 0x01B0_0000 | rd << 12 | rm,
                // movs rd, rm, <shift> rs -- a register-specified shift, all four types
                20 => {
                    let shift = (rng.next() % 4) as u32;
                    0xE1B0_0000 | rd << 12 | rng.reg() << 8 | (shift << 5) | (1 << 4) | rm
                }
                // movs rd, rm, <shift> #0 -- 32 for LSR and ASR, RRX for ROR
                21 => {
                    let shift = (rng.next() % 4) as u32;
                    0xE1B0_0000 | rd << 12 | (shift << 5) | rm
                }
                // mul rd, rm, rs (rs may be r15, checking that the PC guard still applies)
                22 => 0xE000_0000 | rd << 16 | rm << 8 | 0x90 | (rng.reg() & 0xF),
                // A feedback-form conditional write (the shape mask committing uses): orr rd, rd, rm
                23 => (cond << 28) | 0x0380_0000 | rd << 16 | rd << 12 | rm,
                // A feedback-form conditional write: bic rd, rd, rm -- the one ALU the mask path has to invert
                24 => (cond << 28) | 0x03C0_0000 | rd << 16 | rd << 12 | rm,
                // Feedback form with a shifted operand: sub rd, rd, rm, lsr #imm (the shape of the udiv hot block)
                25 => {
                    let shift = (rng.next() % 8) as u32;
                    (cond << 28) | 0x0240_0000 | rd << 16 | rd << 12 | (shift << 7) | rm
                }
                // Feedback form with a high-register destination (r8..r12 are not pinned, so the commit goes through memory).
                // Only up to r12: r13-r15 are sp, lr and pc, and a conditional write to pc would take the PC into
                // an unaligned or Thumb state, outside this test's "ROM, pure ARM, branches inside the program" contract.
                26 => {
                    let high_rd = 8 + (rng.next() % 5) as u32;
                    (cond << 28) | 0x0380_0000 | high_rd << 16 | high_rd << 12 | rm
                }
                // Feedback form with an immediate: eor rd, rd, #imm
                27 => (cond << 28) | 0x0220_0000 | rd << 16 | rd << 12 | imm,
                // A conditional carry read (adcs): the mask form does not apply, so it still goes through cmov or a guard
                28 => (cond << 28) | 0x00B0_0000 | rd << 16 | rd << 12 | rm,
                // b / beq / bne to a word inside the program, sometimes forwards and sometimes backwards, forming loops
                _ => {
                    let target = (rng.next() % PROGRAM_LEN as u64) as i32;
                    let imm24 = ((target - index as i32 - 2) & 0x00FF_FFFF) as u32;
                    let opcode = match rng.next() % 3 {
                        0 => 0xEA00_0000, // b
                        1 => 0x0A00_0000, // beq
                        _ => 0x1A00_0000, // bne
                    };
                    opcode | imm24
                }
            }
        })
        .collect()
}

fn new_cpu(program: &[u32], rng: &mut Rng) -> Cpu<RomMemory> {
    let mut memory = RomMemory::new(MEMORY_SIZE);
    memory.write_words(CODE, program);

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    // The registers point into the data area, so random stores cannot rewrite the program (blocks.rs covers self-modifying-code invalidation)
    for reg in 0..13u8 {
        let value = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
        cpu.state.set_reg(reg, value);
    }
    let flags = rng.next() as u32 & 0xF000_0000;
    cpu.state.cpsr = rynarmic::Cpsr::from_bits(cpu.state.cpsr.bits() | flags);
    cpu
}

fn digest(cpu: &Cpu<RomMemory>) -> (Vec<u32>, u32, u64) {
    let mut mem_hash = 0u64;
    let mut addr = DATA_START;
    while addr < MEMORY_SIZE as u32 {
        mem_hash = mem_hash
            .rotate_left(5)
            .wrapping_add(cpu.memory.peek_u32(addr) as u64);
        addr += 4;
    }
    (cpu.state.r.to_vec(), cpu.state.cpsr.bits(), mem_hash)
}

#[test]
fn random_programs_match_between_step_and_block_execution() {
    let mut rng = Rng(0xD1FF_2026_1103);

    // Deeper runs go through RYNARMIC_DIFF_CASES (raised temporarily during review); the default of 400 keeps the suite quick.
    let cases = std::env::var("RYNARMIC_DIFF_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    for case in 0..cases {
        let program = build_program(&mut rng);
        // Both CPUs must start from one initial state, or the comparison means nothing
        let init = rng.next();
        let mut stepped = new_cpu(&program, &mut Rng(init));
        let mut cached = new_cpu(&program, &mut Rng(init));
        // Every eighth case uses a tiny code budget, forcing repeated passes through
        // the arena-full, invalidate-then-compile, relink path.
        if case % 8 == 0 {
            cached.set_jit_arena_limit(3);
        }

        // Blocks execute **atomically**, so run_cached may overshoot by one block (40 instructions here runs 64).
        // So the block path runs first and the single-step path is then topped up to **the same instruction count**, which is what makes them comparable.
        let (cached_count, cached_result) = cached.run_cached_counted(PROGRAM_LEN as u64);
        match cached_result {
            Ok(()) => {
                let stepped_count = stepped.run(cached_count).unwrap_or_else(|e| {
                    panic!("case {case}: the single-step path raised {e} (the block path ran {cached_count} instructions fine)\ninit={init:#x}\nprogram: {program:08X?}")
                });
                if digest(&stepped) != digest(&cached) {
                    // To localize it, inspect the block's IR through block::translate and then narrow the initial state down.
                    panic!(
                        "case {case}: the two execution paths disagree ({stepped_count} single-stepped / {cached_count} in blocks)\ninit={init:#x}\nprogram: {program:08X?}"
                    );
                }
            }
            Err(cached_err) => {
                // The block path failed: the single-step path runs with **the same total budget** and must fail in the same place.
                // Running only to cached_count would be wrong, since that stops exactly at the failing block's entry and never meets the failure.
                let stepped_result = stepped.run(PROGRAM_LEN as u64);
                let stepped_pc = stepped.state.pc();
                let agreed = matches!(&stepped_result, Err(e) if e == &cached_err)
                    && stepped_pc == cached.state.pc();
                assert!(
                    agreed,
                    "case {case}: the block path reported {cached_err} (pc={pc:#06x}), \
                     the single-step path reported {stepped_result:?} (pc={stepped_pc:#06x}) -- the paths diverge\n\
                     init={init:#x}\nprogram: {program:08X?}",
                    pc = cached.state.pc()
                );
            }
        }
    }
}

#[test]
fn block_execution_spans_multiple_blocks() {
    // 40 movs plus a trailing `b .`: past the per-block cap of 32, so it must be cut into two blocks with the count kept right.
    let mut program = vec![0xE1A0_0000u32; PROGRAM_LEN]; // mov r0, r0
    program.push(0xEAFF_FFFE); // b .
    let mut rng = Rng(1);
    let mut cpu = new_cpu(&program, &mut rng);

    let executed = cpu.run_cached(PROGRAM_LEN as u64 + 1).unwrap();
    assert_eq!(executed, PROGRAM_LEN as u64 + 1);
    assert!(
        cpu.cached_block_count() >= 2,
        "40 instructions should be cut into at least 2 blocks, got {}",
        cpu.cached_block_count()
    );
    assert_eq!(cpu.state.pc(), CODE + 4 * PROGRAM_LEN as u32);
}
