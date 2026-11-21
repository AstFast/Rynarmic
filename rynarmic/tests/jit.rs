//! Consistency tests between the JIT backend and the interpreter.
//!
//! Method: run the same program for the same instruction count on two paths --
//! single-step interpretation and the JIT block cache -- and require registers,
//! CPSR and memory to match bit-for-bit.
//!
//! Two easy-to-miss safeguards:
//!
//! 1. Assert that blocks were actually compiled; otherwise the test can pass
//!    without testing anything if the backend silently gives up on compiling.
//! 2. Replace undefined encodings in random programs with harmless instructions;
//!    decoding is the decoder's concern, not the backend's.

use rynarmic::{Cpsr, Cpu, IrOp, Memory, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
const PROGRAM_LEN: usize = 40;

fn new_cpu(program: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_words(CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu
}

fn thumb_cpu(program: &[u16]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x4000);
    memory.write_thumb(CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(true);
    cpu.state.set_pc(CODE);
    cpu
}

/// xorshift64 with a fixed seed.
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
        (self.next() % 13) as u32
    }

    fn byte(&mut self) -> u32 {
        (self.next() % 256) as u32
    }
}

fn random_state(cpu: &mut Cpu<SimpleMemory>, seed: u64) {
    let mut rng = Rng(seed);
    for index in 0..13u8 {
        // r0-r5 point into the data area so memory accesses hit valid memory; the rest stay random
        let value = if index < 6 {
            DATA + ((rng.next() % 0x1000) as u32 & !3)
        } else {
            rng.next() as u32
        };
        cpu.state.set_reg(index, value);
    }
    let flags = rng.next() as u32 & Cpsr::NZCV;
    cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() | flags);
}

/// Build one random instruction using only families the backend already covers.
fn build_instruction(rng: &mut Rng) -> u32 {
    let rd = rng.reg();
    let rn = rng.reg();
    let rm = rng.reg();
    let rs = rng.reg();
    let cond = if rng.next().is_multiple_of(3) {
        (rng.next() % 14) as u32
    } else {
        0xE
    };

    match rng.next() % 18 {
        // Data processing with immediate and register operands
        0..=8 => {
            let opcode = (rng.next() % 16) as u32;
            let set_flags = u32::from(rng.next().is_multiple_of(2)) << 20;
            if rng.next().is_multiple_of(2) {
                let rotate = (rng.next() % 16) as u32;
                let imm8 = rng.byte();
                (cond << 28)
                    | (1 << 25)
                    | (opcode << 21)
                    | set_flags
                    | (rn << 16)
                    | (rd << 12)
                    | (rotate << 8)
                    | imm8
            } else if rng.next().is_multiple_of(4) {
                // Register-specified shift amount: taken from a random register, so it naturally covers the 0 / 32 / >32 cases
                let shift_type = (rng.next() % 4) as u32;
                (cond << 28)
                    | (opcode << 21)
                    | set_flags
                    | (rn << 16)
                    | (rd << 12)
                    | (rs << 8)
                    | (shift_type << 5)
                    | (1 << 4)
                    | rm
            } else {
                let shift_type = (rng.next() % 4) as u32;
                let shift = (rng.next() % 32) as u32;
                (cond << 28)
                    | (opcode << 21)
                    | set_flags
                    | (rn << 16)
                    | (rd << 12)
                    | (shift << 7)
                    | (shift_type << 5)
                    | rm
            }
        }
        // mul / mla
        9 | 10 => {
            let accumulate = u32::from(rng.next().is_multiple_of(2)) << 21;
            let set_flags = u32::from(rng.next().is_multiple_of(2)) << 20;
            (cond << 28)
                | accumulate
                | set_flags
                | 0x90
                | (rd << 16)
                | (rn << 12)
                | (rm << 8)
                | rng.reg()
        }
        // umull / smull
        11 => {
            let signed = u32::from(rng.next().is_multiple_of(2)) << 22;
            let set_flags = u32::from(rng.next().is_multiple_of(2)) << 20;
            (cond << 28)
                | 0x0080_0090
                | signed
                | set_flags
                | (rd << 16)
                | (rn << 12)
                | (rm << 8)
                | rng.reg()
        }
        // clz
        12 => (cond << 28) | 0x016F_0F10 | (rd << 12) | rm,
        // mrs
        13 => (cond << 28) | 0x010F_0000 | (rd << 12),
        // msr cpsr_f (flags only; avoids changing mode/T)
        14 => (cond << 28) | 0x0120_F000 | (0x8 << 16) | rm,
        // b / bl / bx (the trailing infinite loop catches random targets)
        15 => {
            let offset = rng.byte() & 0x3;
            (cond << 28) | 0x0A00_0000 | offset
        }
        16 => (cond << 28) | 0x0B00_0000 | (rng.byte() & 0x3),
        // ldr rd, [rn, #imm]
        20 => 0xE590_0000 | (rn << 16) | (rd << 12) | rng.byte(),
        // str rd, [rn, #imm]
        21 => 0xE580_0000 | (rn << 16) | (rd << 12) | rng.byte(),
        // ldrb / strb / ldrh / strh / ldrsb / ldrsh
        22 => {
            let op = match rng.next() % 6 {
                0 => 0xE5D0_0000u32, // ldrb
                1 => 0xE5C0_0000,    // strb
                2 => 0xE1D0_00B0,    // ldrh
                3 => 0xE1C0_00B0,    // strh
                4 => 0xE1D0_00D0,    // ldrsb
                _ => 0xE1D0_00F0,    // ldrsh
            };
            op | (rn << 16) | (rd << 12) | rng.byte()
        }
        // ldr rd, [rn], #imm (post-indexed with writeback)
        23 => 0xE490_0000 | (rn << 16) | (rd << 12) | rng.byte(),
        // ldmia / stmia rn!, {rd, ...}
        24 | 25 => {
            let base = 0xE8B0_0000u32
                + if rng.next().is_multiple_of(2) {
                    0
                } else {
                    0x0010_0000
                };
            let count = (rng.next() % 4) as u32 + 1;
            let mut regs = 0u32;
            for bit in 0..count {
                regs |= 1 << ((rd + bit) % 14);
            }
            base | (rn << 16) | regs
        }
        // Unconditional (AL) plain addition, to guarantee forward progress
        _ => 0xE280_0000 | (rn << 16) | (rd << 12) | rng.byte(),
    }
}

fn program(rng: &mut Rng) -> Vec<u32> {
    // Random combinations can produce undefined encodings (for example an illegal
    // shift combination in a data-processing instruction) or hit exceptions such as
    // SWI/BKPT. Neither is the backend's concern: replace the former with a harmless
    // instruction, and the latter is by design "not compiled, handled by the host".
    // Keep the test focused on semantics that can be compiled.
    let mut program: Vec<u32> = (0..PROGRAM_LEN)
        .map(|_| {
            let inst = build_instruction(rng);
            let usable = match rynarmic::decoder::a32::decode(inst, CODE, true) {
                Ok(ops) => !ops
                    .iter()
                    .any(|op| matches!(op, IrOp::Swi { .. } | IrOp::Breakpoint { .. })),
                Err(_) => false,
            };
            if usable { inst } else { 0xE1A0_0000 }
        })
        .collect();
    // End with an infinite loop so execution cannot run past the program
    program[PROGRAM_LEN - 1] = 0xEAFF_FFFE;
    program
}

/// The digest must include memory: a `str` to the wrong address can leave every register identical.
fn digest(cpu: &Cpu<SimpleMemory>) -> ([u32; 16], u32, u64) {
    let mut hash = 0u64;
    let mut addr = DATA;
    while addr < DATA + 0x2000 {
        hash = hash
            .rotate_left(5)
            .wrapping_add(cpu.memory.peek_u32(addr) as u64);
        addr += 4;
    }
    (cpu.state.r, cpu.state.cpsr.bits(), hash)
}

#[test]
fn random_programs_match_between_jit_and_interpreter() {
    let mut rng = Rng(0x5EED_2026_1103);

    for case in 0..400 {
        let program = program(&mut rng);
        let seed = rng.next();

        // Reference path: single-step interpretation (no block cache or JIT)
        let mut reference = new_cpu(&program);
        random_state(&mut reference, seed);
        reference.set_jit_enabled(false);

        // Path under test: block cache + JIT
        let mut jitted = new_cpu(&program);
        random_state(&mut jitted, seed);
        assert!(jitted.jit_enabled());

        // Advance block by block: the JIT runs one block at a time and the reference
        // catches up to the same instruction count, so a mismatch localises the exact
        // range of instructions.
        let mut total = 0u64;
        while total < PROGRAM_LEN as u64 {
            let pre_r = jitted.state.r;
            let pre_cpsr = jitted.state.cpsr.bits();
            let count = jitted.run_cached(1).unwrap();
            total += count;
            reference.run(count).unwrap();

            if digest(&reference) != digest(&jitted) {
                let offset = total as usize - count as usize;
                panic!(
                    "case {case}: mismatch after instructions {offset}..{total}\n\
                     mismatched block: {:08X?}\npre-block state r={pre_r:08X?} cpsr={pre_cpsr:#010X}\nwhole program: {program:08X?}\n\
                     reference r={:08X?} cpsr={:#010X}\n\
                     JIT  r={:08X?} cpsr={:#010X}",
                    &program[offset..(total as usize).min(program.len())],
                    reference.state.r,
                    reference.state.cpsr.bits(),
                    jitted.state.r,
                    jitted.state.cpsr.bits(),
                );
            }
        }
    }
}

/// Compare a memory instruction: registers, CPSR and data-area contents must all match.
fn check_memory_instruction(instruction: u32, setup: impl Fn(&mut Cpu<SimpleMemory>) + Copy) {
    let run = |jit: bool| {
        let mut memory = SimpleMemory::new(0, 0x8000);
        memory.write_words(CODE, &[instruction, 0xEAFF_FFFE]);
        for index in 0..24u32 {
            memory.write_u32(DATA + index * 4, 0x1111_0000 + index);
        }
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        setup(&mut cpu);
        cpu.set_jit_enabled(jit);
        if jit {
            cpu.run_cached(1).unwrap();
            // Prevent this comparison silently degrading into two interpreted paths when no block was compiled
            assert!(
                cpu.compiled_block_count() > 0,
                "memory instruction {instruction:#010X} should be JIT compiled"
            );
        } else {
            cpu.run(1).unwrap();
        }
        let data: Vec<u32> = (0..24)
            .map(|index| cpu.memory.peek_u32(DATA + index * 4))
            .collect();
        (cpu.state.r, cpu.state.cpsr.bits(), data)
    };

    let reference = run(false);
    let jitted = run(true);
    assert_eq!(
        reference, jitted,
        "memory instruction {instruction:#010X}: JIT and interpreter disagree"
    );
}

#[test]
fn load_store_variants_match() {
    for instruction in [
        0xE591_0000, // ldr   r0, [r1]
        0xE581_0000, // str   r0, [r1]
        0xE5D1_0003, // ldrb  r0, [r1, #3]
        0xE5C1_0003, // strb  r0, [r1, #3]
        0xE1D1_00B2, // ldrh  r0, [r1, #2]
        0xE1C1_00B2, // strh  r0, [r1, #2]
        0xE1D1_00D1, // ldrsb r0, [r1, #1]
        0xE1D1_00F2, // ldrsh r0, [r1, #2]
        0xE5B1_0008, // ldr   r0, [r1, #8]!
        0xE491_0008, // ldr   r0, [r1], #8
        0xE521_0004, // str   r0, [r1, #-4]!
        0xE601_0005, // str   r0, [r1], -r5
        0xE591_0002, // ldr   r0, [r1, r2]
        0xE791_0102, // ldr   r0, [r1, r2, lsl #2]
    ] {
        check_memory_instruction(instruction, |cpu| {
            cpu.state.set_reg(0, 0xDEAD_BEEF);
            cpu.state.set_reg(1, DATA + 16);
            cpu.state.set_reg(2, 4);
            cpu.state.set_reg(5, 8);
        });
        // A second set of values covers negative offsets and unaligned addresses
        check_memory_instruction(instruction, |cpu| {
            cpu.state.set_reg(0, 0x0000_7F80);
            cpu.state.set_reg(1, DATA + 40);
            cpu.state.set_reg(2, 12);
            cpu.state.set_reg(5, 16);
        });
    }
}

#[test]
fn ldm_stm_variants_match() {
    for instruction in [
        0xE8B1_000C, // ldmia r1!, {r2, r3}
        0xE8A1_000C, // stmia r1!, {r2, r3}
        0xE931_000C, // ldmdb r1!, {r2, r3}
        0xE921_000C, // stmdb r1!, {r2, r3}
        0xE891_000C, // ldmia r1, {r2, r3} (no writeback)
        0xE981_0042, // stmib r1, {r1, r6}
        0xE92D_4010, // push {r4, lr}
        0xE8B1_8001, // ldmia r1!, {r0, pc} -- PC is in the register list, so the block ends with a branch
    ] {
        check_memory_instruction(instruction, |cpu| {
            cpu.state.set_reg(0, 0xA5A5_0001);
            cpu.state.set_reg(1, DATA + 32);
            cpu.state.set_reg(2, 0x2222_0002);
            cpu.state.set_reg(3, 0x3333_0003);
            cpu.state.set_reg(4, 0x4444_0004);
            cpu.state.set_reg(6, 0x6666_0006);
            cpu.state.set_reg(13, DATA + 64);
            cpu.state.set_reg(14, 0x7777_0007);
        });
    }
}

#[test]
fn jit_compiles_memory_blocks() {
    // Memory blocks must really be compiled, otherwise the comparisons above only compare two interpreted paths
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(
        CODE,
        &[
            0xE591_0000, // ldr r0, [r1]
            0xE581_0004, // str r0, [r1, #4]
            0xEAFF_FFFE, // b .
        ],
    );
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(1, DATA);
    cpu.run_cached(4).unwrap();
    assert!(
        cpu.compiled_block_count() > 0,
        "a block containing memory accesses should be compiled"
    );
}

#[test]
fn jit_actually_compiles_blocks() {
    // Prevent the previous test from passing vacuously when nothing was compiled
    let program = vec![
        0xE3A0_0001, // mov r0, #1
        0xE1A0_0002, // mov r1, r2 expands to an ordinary instruction
        0xE081_1000, // add r1, r1, r0
        0xEAFF_FFFE, // b .
    ];
    let mut cpu = new_cpu(&program);
    cpu.run_cached(8).unwrap();
    assert!(
        cpu.compiled_block_count() > 0,
        "arithmetic block should be compiled to machine code (got {} blocks)",
        cpu.compiled_block_count()
    );
}

/// A unit-vector style check: given initial registers/flags, run one instruction and assert the result.
///
/// The JIT and the interpreter must produce the same `(registers, CPSR)`.
fn check_single(instruction: u32, init: impl Fn(&mut Cpu<SimpleMemory>)) -> ([u32; 16], u32, u64) {
    let mut reference = new_cpu(&[instruction, 0xEAFF_FFFE]);
    init(&mut reference);
    reference.set_jit_enabled(false);
    reference.run(1).unwrap();

    let mut jitted = new_cpu(&[instruction, 0xEAFF_FFFE]);
    init(&mut jitted);
    jitted.run_cached(1).unwrap();

    assert_eq!(
        digest(&reference),
        digest(&jitted),
        "instruction {instruction:#010X}: JIT and interpreter disagree"
    );
    digest(&jitted)
}

#[test]
fn arithmetic_carry_and_overflow_match() {
    // For subtraction C is the inverse of the borrow -- the biggest x86 vs ARM difference
    for (a, b) in [
        (0u32, 1u32),
        (1, 1),
        (0x8000_0000, 1),
        (0x7FFF_FFFF, 0xFFFF_FFFF),
        (0xFFFF_FFFF, 0xFFFF_FFFF),
    ] {
        // subs r0, r1, r2
        let (r, _, _) = check_single(0xE051_0002, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
        });
        assert_eq!(r[0], a.wrapping_sub(b));

        // adds r0, r1, r2
        let (r, _, _) = check_single(0xE091_0002, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
        });
        assert_eq!(r[0], a.wrapping_add(b));

        // sbcs / adcs read C
        check_single(0xE0D1_0002, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
            cpu.state.cpsr.set_c(true);
        });
        check_single(0xE0B1_0002, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
            cpu.state.cpsr.set_c(true);
        });
        // rsbs
        check_single(0xE071_0002, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
        });
    }
}

#[test]
fn shifted_operand_carry_matches() {
    for amount in [0u32, 1, 5, 31] {
        for shift_type in 0..4u32 {
            // movs r0, r1, <shift> #amount
            let instruction = 0xE1B0_0000 | (amount << 7) | (shift_type << 5) | 1;
            check_single(instruction, |cpu| {
                cpu.state.set_reg(1, 0x8000_0001);
                cpu.state.cpsr.set_c(true);
            });
            check_single(instruction, |cpu| {
                cpu.state.set_reg(1, 0x1234_5678);
                cpu.state.cpsr.set_c(false);
            });
        }
    }
}

#[test]
fn logical_ops_keep_overflow_and_carry_rules() {
    for opcode in [0x0u32, 0x1, 0xC, 0xE, 0xF] {
        // ands / eors / orrs / bics / mvns, with a shift
        let instruction = 0xE010_0000 | (opcode << 21) | (1 << 20) | (1 << 16) | (5 << 7) | 1;
        check_single(instruction, |cpu| {
            cpu.state.set_reg(1, 0xF000_0003);
            cpu.state.set_reg(1, 0xF000_0003);
            cpu.state.cpsr = Cpsr::from_bits(Cpsr::V | Cpsr::C);
        });
    }
}

#[test]
fn mul_and_clz_match() {
    for (a, b) in [
        (0u32, 0u32),
        (3, 7),
        (0xFFFF_FFFF, 2),
        (0x8000_0000, 0x8000_0000),
    ] {
        // muls r0, r1, r2
        check_single(0xE010_0291, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
        });
        // umull r0, r1, r2, r3
        check_single(0xE081_0392, |cpu| {
            cpu.state.set_reg(2, a);
            cpu.state.set_reg(3, b);
        });
        // smull r0, r1, r2, r3
        check_single(0xE0C1_0392, |cpu| {
            cpu.state.set_reg(2, a);
            cpu.state.set_reg(3, b);
        });
    }

    for value in [0u32, 1, 0x8000_0000, 0x00FF_FFFF] {
        check_single(0xE16F_0F11, |cpu| cpu.state.set_reg(1, value));
    }
}

#[test]
fn conditional_execution_matches() {
    // moveq r0, #7 / addne r0, r1, #3 / movsne r2, r3
    for cond in 0..14u32 {
        for (n, z, c, v) in [
            (false, false, false, false),
            (false, true, false, false),
            (true, false, true, false),
            (false, false, false, true),
            (true, true, true, true),
        ] {
            let instruction = (cond << 28) | 0x03A0_0007;
            check_single(instruction, |cpu| {
                cpu.state.cpsr = Cpsr::from_bits(
                    Cpsr::pack_nzcv(if n { 0x8000_0000 } else { 1 }, c, v)
                        | if z { Cpsr::Z } else { 0 },
                );
            });

            let instruction = (cond << 28) | 0x0281_0003;
            check_single(instruction, |cpu| {
                cpu.state.set_reg(1, 0x1234_5678);
                cpu.state.cpsr = Cpsr::from_bits(
                    Cpsr::pack_nzcv(if n { 0x8000_0000 } else { 1 }, c, v)
                        | if z { Cpsr::Z } else { 0 },
                );
            });
        }
    }
}

#[test]
fn branch_and_exchange_match() {
    // bne to 0x1020 / bx lr / mov pc, lr
    for cond in 0..14u32 {
        check_single((cond << 28) | 0x0A00_0006, |cpu| {
            cpu.state.cpsr.set_z(false);
        });
        check_single((cond << 28) | 0x0A00_0006, |cpu| {
            cpu.state.cpsr.set_z(true);
        });
    }

    check_single(0xE12F_FF1E, |cpu| {
        cpu.state.set_reg(14, 0x1011);
        cpu.state.cpsr.set_t(false);
    });
}

#[test]
fn status_register_transfer_matches() {
    // mrs r0, cpsr
    check_single(0xE10F_0000, |cpu| {
        cpu.state.cpsr = Cpsr::from_bits(0xF000_00D3);
    });
    // msr cpsr_f, r1
    check_single(0xE129_F001, |cpu| {
        cpu.state.set_reg(1, 0xFFFF_FFFF);
        cpu.state.cpsr = Cpsr::from_bits(0x0000_00D3);
    });
}

#[test]
fn shift_zero_and_rrx_match_jit() {
    for inst in [0xE1B0_0021u32, 0xE1B0_0041, 0xE1B0_0061, 0xE1B0_0001] {
        for carry in [false, true] {
            let mut stepped = new_cpu(&[inst, 0xEAFF_FFFE]);
            let mut cached = new_cpu(&[inst, 0xEAFF_FFFE]);
            for cpu in [&mut stepped, &mut cached] {
                cpu.state.set_reg(1, 0x8000_0001);
                let bits = (cpu.state.cpsr.bits() & !Cpsr::C) | if carry { Cpsr::C } else { 0 };
                cpu.state.cpsr = Cpsr::from_bits(bits);
            }

            stepped.run(1).unwrap();
            cached.run_cached(1).unwrap();

            assert_eq!(stepped.state.r, cached.state.r, "inst={inst:#010x}");
            assert_eq!(
                stepped.state.cpsr.bits(),
                cached.state.cpsr.bits(),
                "inst={inst:#010x}"
            );
            assert!(
                cached.compiled_block_count() > 0,
                "inst={inst:#010x} should be JIT compiled"
            );
        }
    }
}

#[test]
fn a32_register_shifts_match_block_path() {
    for inst in [0xE1B0_0231u32, 0xE1B0_0251, 0xE1B0_0271] {
        let mut stepped = new_cpu(&[inst, 0xEAFF_FFFE]);
        let mut cached = new_cpu(&[inst, 0xEAFF_FFFE]);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(1, 0x8000_0001);
            cpu.state.set_reg(2, 3);
            cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::C);
        }

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.state.r, cached.state.r, "inst={inst:#010x}");
        assert_eq!(
            stepped.state.cpsr.bits(),
            cached.state.cpsr.bits(),
            "inst={inst:#010x}"
        );
    }
}

#[test]
fn thumb_shift_by_32_matches_jit() {
    for inst in [0x0800u16, 0x1000] {
        let mut stepped = thumb_cpu(&[inst, 0xE7FE]);
        let mut cached = thumb_cpu(&[inst, 0xE7FE]);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(0, 0x8000_0001);
        }

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.state.r, cached.state.r, "inst={inst:#06x}");
        assert_eq!(
            stepped.state.cpsr.bits(),
            cached.state.cpsr.bits(),
            "inst={inst:#06x}"
        );
        assert!(
            cached.compiled_block_count() > 0,
            "inst={inst:#06x} should be JIT compiled"
        );
    }
}

#[test]
fn load_register_shift_matches_jit() {
    // ldr r0, [r1, r2, <shift>]: the JIT's address calculation must match the interpreter
    for (inst, r2, carry) in [
        (0xE791_0122u32, 16u32, false), // lsr #2 -> offset 4
        (0xE791_0022, 1, false),        // lsr #0 == #32 -> offset 0
        (0xE791_0062, 0, true),         // ror #0 == rrx(C=1) -> offset 0x80000000
    ] {
        let mut stepped = new_cpu(&[inst, 0xEAFF_FFFE]);
        let mut cached = new_cpu(&[inst, 0xEAFF_FFFE]);
        for cpu in [&mut stepped, &mut cached] {
            cpu.state.set_reg(1, DATA);
            cpu.state.set_reg(2, r2);
            cpu.state.cpsr = Cpsr::from_bits(
                (cpu.state.cpsr.bits() & !Cpsr::C) | if carry { Cpsr::C } else { 0 },
            );
            cpu.memory.write_words(DATA, &[0x1234_5678, 0x9ABC_DEF0]);
        }

        stepped.run(1).unwrap();
        cached.run_cached(1).unwrap();

        assert_eq!(stepped.state.r, cached.state.r, "inst={inst:#010x}");
        assert_eq!(
            stepped.state.cpsr.bits(),
            cached.state.cpsr.bits(),
            "inst={inst:#010x}"
        );
        assert!(
            cached.compiled_block_count() > 0,
            "inst={inst:#010x} should be JIT compiled"
        );
    }
}

/// Seed the initial state for cross-block cases: NZCV is deliberately inconsistent
/// with any arithmetic result, so both "materialisation did not really overwrite
/// CPSR" and "the descriptor was saved incorrectly" become visible.
fn seed_lazy(cpu: &mut Cpu<SimpleMemory>, r0: u32) {
    cpu.state.r[0] = r0;
    cpu.state.r[1] = 0xFFFF_FFFF;
    cpu.state.r[2] = 0x8000_0001;
    cpu.state.r[3] = 7;
    cpu.state.r[4] = DATA;
    cpu.state.r[5] = DATA + 0x40;
    cpu.state.r[6] = 0xA5A5_1234;
    cpu.state.r[9] = 1;
    cpu.state.cpsr.set_n(true);
    cpu.state.cpsr.set_z(true);
    cpu.state.cpsr.set_c(false);
    cpu.state.cpsr.set_v(true);
}

/// Cross-block lazy flags.
///
/// Block-by-block comparison (`run_cached(1)`) materialises at every block boundary,
/// which exactly hides the "descriptor survives across blocks" path. Here the JIT
/// runs several blocks in one call and only materialises when the call returns.
#[test]
fn cross_block_lazy_flags_match_interpreter() {
    let cases: &[(&str, &[u32], u32)] = &[
        (
            "adcs reads C left by the previous block",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xEAFF_FFFF, // b 0x1008
                0xE2B1_1000, // adcs r1, r1, #0
                0xEAFF_FFFB, // b 0x1000
            ],
            4,
        ),
        (
            // HI after add cannot be expressed with one setcc, so the conditional branch must fall back to materialisation
            "bhi falls back on an add-like operation",
            &[
                0xE090_0001, // adds r0, r0, r1
                0x8A00_0001, // bhi 0x1010
                0xE243_3001, // sub r3, r3, #1
                0xEAFF_FFFE, // b .
                0xE284_4002, // add r4, r4, #2
                0xE282_2001, // add r2, r2, #1
                0xE350_0064, // cmp r0, #100
                0x9AFF_FFF7, // bls 0x1000
            ],
            4,
        ),
        (
            "conditional branch reads the descriptor after a memory callback",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xE594_4000, // ldr r4, [r4]
                0xE585_6000, // str r6, [r5]
                0x1AFF_FFFB, // bne 0x1000
                0xEAFF_FFFE, // b .
            ],
            4,
        ),
        (
            "RRX reads C after lazy flags",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xE1B0_1062, // movs r1, r2, rrx
                0xEAFF_FFFC, // b 0x1000
            ],
            4,
        ),
        (
            "a flag-preserving mul between lazy flags and a conditional branch",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xE000_0099, // mul r0, r9, r0
                0x1A00_0000, // bne 0x1010
                0xE251_1001, // subs r1, r1, #1
                0xE280_0001, // add r0, r0, #1
                0xE250_0001, // subs r0, r0, #1
                0xE000_0099, // mul r0, r9, r0
                0x0A00_0002, // beq 0x102c
                0xE243_3001, // sub r3, r3, #1
                0xEAFF_FFF5, // b 0x1000
                0xE250_0001, // subs r0, r0, #1
                0xE2B1_1000, // adcs r1, r1, #0
                0xEAFF_FFFC, // b 0x1028
            ],
            4,
        ),
        (
            // The middle block neither reads nor writes flags (EntryFlags::NoAccess); the descriptor must pass through unchanged
            "descriptor crosses a block that never touches flags",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xEA00_0001, // b 0x1010
                0xE282_2001, // add r2, r2, #1 (unreachable)
                0xE282_2001, // add r2, r2, #1 (unreachable)
                0xE283_3002, // add r3, r3, #2
                0xE284_4003, // add r4, r4, #3
                0xEA00_0000, // b 0x1020
                0xE285_5001, // add r5, r5, #1 (unreachable)
                0x1AFF_FFF6, // bne 0x1000
                0xEAFF_FFF5, // b 0x1000
            ],
            4,
        ),
        (
            // Entry Discards: the previous block's C must really be dropped
            "entry discards a stale descriptor",
            &[
                0xE250_0001, // subs r0, r0, #1        C=1
                0xEAFF_FFFF, // b 0x1008
                0xE251_1002, // subs r1, r1, #2        overwrites NZCV
                0xEAFF_FFFF, // b 0x1010
                0xE2B2_2000, // adcs r2, r2, #0
                0xE2B3_3000, // adcs r3, r3, #0
                0xEAFF_FFFA, // b 0x1008
            ],
            4,
        ),
        (
            "a cross-block descriptor is read as a whole word by MRS",
            &[
                0xE250_0001, // subs r0, r0, #1
                0xEAFF_FFFF, // b 0x1008
                0xE10F_2000, // mrs r2, cpsr
                0xE251_1001, // subs r1, r1, #1
                0xE152_0001, // cmp r2, r1
                0x1A00_0001, // bne 0x1020
                0xEAFF_FFFA, // b 0x1008
                0xE285_5001, // add r5, r5, #1 (unreachable)
                0xE243_3001, // sub r3, r3, #1
                0xEAFF_FFF5, // b 0x1000
            ],
            4,
        ),
        (
            // Both the callback and the address addition clobber x86 CF, so C must be saved into the descriptor before that happens
            "C survives a memory callback",
            &[
                0xE250_0001, // subs r0, r0, #1   C=1
                0xE594_4000, // ldr r4, [r4]
                0xE585_6000, // str r6, [r5]
                0xE2B1_1000, // adcs r1, r1, #0   must add 1
                0xEAFF_FFFA, // b 0x1000
            ],
            4,
        ),
        (
            // Likewise V must be saved into the descriptor before add clobbers x86 OF
            "V survives a flag-clobbering add",
            &[
                0xE250_0001, // subs r0, r0, #1    r0=INT_MIN -> V=1
                0xE282_2001, // add r2, r2, #1     OF=0
                0x6A00_0001, // bvs 0x1014
                0xEAFF_FFFB, // b 0x1000
                0xE283_3001, // add r3, r3, #1 (unreachable)
                0xE284_4002, // add r4, r4, #2     <- only reached when V=1
                0xEAFF_FFF8, // b 0x1000
            ],
            0x8000_0000,
        ),
    ];

    for (name, program, r0) in cases {
        let mut jitted = new_cpu(program);
        seed_lazy(&mut jitted, *r0);
        let mut reference = new_cpu(program);
        seed_lazy(&mut reference, *r0);

        let mut executed = 0u64;
        // Run several blocks in one call: no boundary materialisation via run_cached between blocks
        while executed < 400 {
            executed += jitted.run_cached(200).unwrap();
        }
        reference.run(executed).unwrap();

        assert!(
            jitted.compiled_block_count() > 0,
            "{name}: blocks should be JIT compiled, otherwise nothing was tested"
        );
        assert_eq!(digest(&reference), digest(&jitted), "{name}");
    }
}

/// Return the decoded IR family name to confirm the test encodings are not mistyped:
/// with a wrong encoding both paths run the same wrong instruction and the
/// comparison becomes vacuous.
fn decoded_family(inst: u32) -> &'static str {
    match rynarmic::decoder::a32::decode(inst, CODE, true) {
        Ok(ops) => match ops.first() {
            Some(IrOp::Saturating { .. }) => "saturating",
            Some(IrOp::MulHalf { .. }) => "mulhalf",
            Some(IrOp::MulWordHalf { .. }) => "mulwordhalf",
            Some(IrOp::Swap { .. }) => "swap",
            Some(IrOp::LoadDouble { .. }) => "loaddouble",
            Some(IrOp::StoreDouble { .. }) => "storedouble",
            other => panic!("{inst:#010X} decoded as {other:?}"),
        },
        Err(err) => panic!("{inst:#010X} failed to decode: {err}"),
    }
}

/// The JIT and interpreter must agree, and the block must **really be compiled**.
fn check_single_compiled(
    instruction: u32,
    init: impl Fn(&mut Cpu<SimpleMemory>),
) -> ([u32; 16], u32, u64) {
    let mut reference = new_cpu(&[instruction, 0xEAFF_FFFE]);
    init(&mut reference);
    reference.set_jit_enabled(false);
    reference.run(1).unwrap();

    let mut jitted = new_cpu(&[instruction, 0xEAFF_FFFE]);
    init(&mut jitted);
    jitted.run_cached(1).unwrap();

    assert!(
        jitted.compiled_block_count() > 0,
        "instruction {instruction:#010X} should be JIT compiled"
    );
    assert_eq!(
        digest(&reference),
        digest(&jitted),
        "instruction {instruction:#010X}: JIT and interpreter disagree"
    );
    digest(&jitted)
}

/// JIT comparisons for v5TE saturating arithmetic, halfword multiplies, doubleword accesses and atomic swaps.
#[test]
fn v5te_families_match_jit() {
    // Self-check the encodings first: one wrong digit decodes to a different instruction
    for (inst, family) in [
        (0xE100_0051u32, "saturating"), // qadd   r0, r1, r2
        (0xE120_0051, "saturating"),    // qsub   r0, r1, r2
        (0xE140_0051, "saturating"),    // qdadd  r0, r1, r2
        (0xE160_0051, "saturating"),    // qdsub  r0, r1, r2
        (0xE160_0182, "mulhalf"),       // smulbb r0, r1, r2
        (0xE103_0182, "mulhalf"),       // smlabb r0, r1, r2, r3
        (0xE141_0283, "mulhalf"),       // smlalbb r2, r3, ... (RdLo=r2)
        (0xE120_01A2, "mulwordhalf"),   // smulwb r0, r1, r2
        (0xE123_0182, "mulwordhalf"),   // smlawb r0, r1, r2, r3
        (0xE102_0091, "swap"),          // swp    r0, r1, [r2]
        (0xE142_0091, "swap"),          // swpb   r0, r1, [r2]
        (0xE1C1_20D8, "loaddouble"),    // ldrd   r2, [r1, #8]
        (0xE1C1_00F8, "storedouble"),   // strd   r0, [r1, #8]
    ] {
        assert_eq!(decoded_family(inst), family, "encoding {inst:#010X}");
    }

    // Saturating: cover both bounds, saturation after doubling, and the non-saturating common path
    for inst in [0xE100_0051u32, 0xE120_0051, 0xE140_0051, 0xE160_0051] {
        for (rm, rn) in [
            (0x7FFF_FFFFu32, 1u32),
            (0x8000_0000, 0xFFFF_FFFF),
            (0x4000_0000, 0x4000_0000),
            (5, 7),
            (0xFFFF_FFFD, 2),
            (0x7FFF_FFFF, 0x7FFF_FFFF),
        ] {
            check_single_compiled(inst, |cpu| {
                cpu.state.set_reg(1, rm);
                cpu.state.set_reg(2, rn);
                cpu.state.cpsr.set_q(false);
            });
        }
    }
    // Q is a sticky set-only bit: set it, then run a non-saturating instruction; it must not be cleared
    let (_, cpsr, _) = check_single_compiled(0xE100_0051, |cpu| {
        cpu.state.set_reg(1, 1);
        cpu.state.set_reg(2, 2);
        cpu.state.cpsr.set_q(true);
    });
    assert!(
        Cpsr::from_bits(cpsr).q(),
        "a non-saturating execution must not clear Q"
    );

    // Halfword multiplies: exercise all four halfword combinations
    for inst in [
        0xE160_0182u32, // smulbb
        0xE160_01C2,    // smulbt
        0xE160_01A2,    // smultb
        0xE160_01E2,    // smultt
        0xE103_0182,    // smlabb
        0xE141_0283,    // smlalbb
        0xE120_01A2,    // smulwb
        0xE123_0182,    // smlawb
    ] {
        for (r1, r2, r3) in [
            (0x0001_0002u32, 0x0003_0004u32, 0x0000_0005u32),
            (0x8001_FFFF, 0x8002_0003, 0xFFFF_FFF0),
            (0x7FFF_7FFF, 0x7FFF_7FFF, 0x0000_1000),
            (0xFFFF_FFFF, 0xFFFF_FFFF, 1),
        ] {
            check_single_compiled(inst, |cpu| {
                cpu.state.set_reg(1, r1);
                cpu.state.set_reg(2, r2);
                cpu.state.set_reg(3, r3);
                cpu.state.set_reg(4, 0);
                cpu.state.set_reg(5, 0);
                // Halfword multiplies write only N/Z; C/V must be preserved unchanged
                cpu.state.cpsr = Cpsr::from_bits(Cpsr::C | Cpsr::V);
            });
        }
    }

    // Accumulation overflow in SMLAxy / SMLAWy must set Q; SMULxy must not touch Q.
    // The product is only 30 bits, so pushing Ra near INT32_MAX overflows the addition.
    for (inst, a, b, acc) in [
        // smlabb r0, r1, r2, r3: 32767 x 32767 + 0x7FFFFFFF
        (
            0xE100_3182u32,
            0x0000_7FFFu32,
            0x0000_7FFFu32,
            0x7FFF_FFFFu32,
        ),
        // smlawb r0, r1, r2, r3: full word in r2 (bits 3-0), y halfword in r1 (bits 11-8)
        // (observed with a unicorn/qemu probe): (0x7FFFFFFF x 0x7FFF) >> 16 + 0x7FFFFFFF
        (0xE120_3182, 0x0000_7FFF, 0x7FFF_FFFF, 0x7FFF_FFFF),
    ] {
        let (_, cpsr, _) = check_single_compiled(inst, |cpu| {
            cpu.state.set_reg(1, a);
            cpu.state.set_reg(2, b);
            cpu.state.set_reg(3, acc);
            cpu.state.cpsr.set_q(false);
        });
        assert!(
            Cpsr::from_bits(cpsr).q(),
            "{inst:#010X}: accumulation overflow must set Q"
        );
    }
    let (_, cpsr, _) = check_single_compiled(0xE160_0182, |cpu| {
        cpu.state.set_reg(1, 0x0000_7FFF);
        cpu.state.set_reg(2, 0x0000_7FFF);
        cpu.state.cpsr.set_q(false);
    });
    assert!(
        !Cpsr::from_bits(cpsr).q(),
        "SMULbb has no accumulation and must not touch Q"
    );

    // Doubleword accesses and atomic swaps operate on real memory
    for instruction in [
        0xE1C1_00F8u32, // strd r0, [r1, #8]
        0xE1C1_20D8,    // ldrd r2, [r1, #8]
        0xE1C1_20F8,    // strd r2, [r1, #8]
        0xE1C1_00D8,    // ldrd r0, [r1, #8]
        0xE1A1_00F8,    // strd r0, [r1, #8]! (register offset, writeback)
        0xE1A1_20D8,    // ldrd r2, [r1, #8]!
        0xE1E1_00F8,    // strd r0, [r1, #8]! -- base r1 is one of the pair being stored
        0xE1F1_10D8,    // ldrd r1, [r1, #8]! -- destination overlaps the base register
        0xE102_0091,    // swp  r0, r1, [r2]
        0xE142_0091,    // swpb r0, r1, [r2]
    ] {
        check_memory_instruction(instruction, |cpu| {
            cpu.state.set_reg(0, 0x1111_2222);
            cpu.state.set_reg(1, DATA + 0x10);
            cpu.state.set_reg(2, 0x3333_4444);
            cpu.state.set_reg(3, 0x5555_6666);
            cpu.state.cpsr = Cpsr::from_bits(Cpsr::C);
        });
    }
}

/// Register-specified shift amount (`movs r0, r1, lsl r2`).
///
/// Every trap on this path is at a boundary: shift amounts 0 (value unchanged and
/// **C unchanged**), 32 and >32, plus x86 truncating a variable shift count to 5 bits.
/// So 0/1/31/32/33/64/255 are all exercised here.
#[test]
fn register_shift_matches_jit() {
    // movs r0, r1, <shift> r2 (four shift types)
    let logical = [0xE1B0_0211u32, 0xE1B0_0231, 0xE1B0_0251, 0xE1B0_0271];
    // adds r0, r1, r2, lsl r3 / ands r0, r1, r2, lsr r3
    let arithmetic = [0xE091_0312u32, 0xE011_0332];

    for instruction in logical.into_iter().chain(arithmetic) {
        for amount in [0u32, 1, 5, 31, 32, 33, 64, 255] {
            for value in [0x8000_0001u32, 0x1234_5678, 0xFFFF_FFFF] {
                for carry in [false, true] {
                    check_single_compiled(instruction, |cpu| {
                        cpu.state.set_reg(1, value);
                        cpu.state.set_reg(2, amount);
                        cpu.state.set_reg(3, amount);
                        let bits =
                            (cpu.state.cpsr.bits() & !Cpsr::C) | if carry { Cpsr::C } else { 0 };
                        cpu.state.cpsr = Cpsr::from_bits(bits);
                    });
                }
            }
        }
    }
}

/// Cross-block lazy flags for Thumb: Thumb block-boundary rules differ from A32
/// (every conditional branch ends a block), which the A32 cases above do not cover.
#[test]
fn thumb_cross_block_lazy_flags_match_interpreter() {
    // subs r1, #1 (block 1) -> b -> adcs r2, r2 (block 2, reads C from block 1) -> b back to block 1
    let program = [
        0x1E49, // subs r1, r1, #1
        0xE001, // b 0x1008
        0xBF00, // nop (unreachable)
        0xBF00, // nop (unreachable)
        0x4152, // adcs r2, r2
        0xE7F9, // b 0x1000
    ];

    let mut jitted = thumb_cpu(&program);
    let mut reference = thumb_cpu(&program);
    for cpu in [&mut jitted, &mut reference] {
        cpu.state.set_reg(1, 4);
        cpu.state.set_reg(2, 1);
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() & !Cpsr::NZCV);
    }

    let mut executed = 0u64;
    while executed < 200 {
        executed += jitted.run_cached(50).unwrap();
    }
    reference.run(executed).unwrap();

    assert!(
        jitted.compiled_block_count() > 0,
        "blocks should be JIT compiled"
    );
    // Sanity tooth: the loop really ran (adcs doubles r2 and adds C each iteration)
    assert!(jitted.state.reg(2) > 1, "adcs should execute repeatedly");

    assert_eq!(digest(&reference), digest(&jitted));
}

/// `POP {…, pc}` must put SP back exactly where `PUSH` left it. A Thumb block that returns
/// through PC ends with a dynamic jump, and the writeback that goes with the load is the part a
/// register-list transfer can silently lose: the caller then runs on a stack sixteen bytes low,
/// which is what a real pack showed as a return address of zero two frames later.
#[test]
fn thumb_push_pop_with_pc_restores_sp_under_jit() {
    // push {r4, r5, r6, lr} ; movs r0, #9 ; pop {r4, r5, r6, pc}
    let program = [0xB570u16, 0x2009, 0xBD70];
    let spin = 0x1200u32;
    let setup = |cpu: &mut Cpu<SimpleMemory>| {
        cpu.memory.write_thumb(spin, &[0xE7FE]);
        cpu.state.set_reg(13, 0x3000);
        cpu.state.set_reg(14, spin | 1);
        cpu.state.set_reg(4, 0x4444_4444);
        cpu.state.set_reg(5, 0x5555_5555);
        cpu.state.set_reg(6, 0x6666_6666);
    };

    let mut reference = thumb_cpu(&program);
    setup(&mut reference);
    reference.run(3).unwrap();

    let mut jitted = thumb_cpu(&program);
    setup(&mut jitted);
    jitted.run_cached(3).unwrap();
    assert!(
        jitted.compiled_block_count() > 0,
        "the block should be JIT compiled"
    );
    assert_eq!(digest(&reference), digest(&jitted));
}

/// `BLX Rm` is a call, not a plain branch: it writes LR, and a callee that returns through
/// `pop {…, pc}` depends on that. A decoder that treats it as `BX` sends the return two frames
/// up and leaves the callee's pushed frame behind — the shape a real pack showed as a zero PC.
#[test]
fn thumb_blx_register_links_under_jit() {
    let program = [
        0x2005u16, // 1000: movs r0, #5
        0x4788,    // 1002: blx r1
        0x4602,    // 1004: mov r2, r0
        0xE7FE,    // 1006: b .
    ];
    let setup = |cpu: &mut Cpu<SimpleMemory>| {
        cpu.memory.write_thumb(
            0x1100,
            &[
                0xB530, // push {r4, r5, lr}
                0x2401, // movs r4, #1
                0x1900, // adds r0, r0, r4
                0xBD30, // pop {r4, r5, pc}
            ],
        );
        cpu.memory.write_thumb(0x1200, &[0xE7FE]);
        cpu.state.set_reg(1, 0x1101);
        cpu.state.set_reg(13, 0x3000);
        cpu.state.set_reg(14, 0x1201);
    };

    let mut reference = thumb_cpu(&program);
    setup(&mut reference);
    reference.run(8).unwrap();

    let mut jitted = thumb_cpu(&program);
    setup(&mut jitted);
    jitted.run_cached(8).unwrap();
    assert!(
        jitted.compiled_block_count() > 0,
        "the block should be JIT compiled"
    );
    assert_eq!(digest(&reference), digest(&jitted));
}

/// The ARMv6 Thumb extends are expressed as two ordinary shifts, so the JIT has to agree with the
/// interpreter for all four — a pack built for a v6 handset executes them on its start-up path.
#[test]
fn thumb_armv6_extends_match_jit() {
    for (insn, input) in [
        (0xB288u16, 0x0000_8001u32), // uxth r0, r1
        (0xB2C8, 0x0000_00ff),       // uxtb r0, r1
        (0xB208, 0x0000_8001),       // sxth r0, r1
        (0xB248, 0x0000_0080),       // sxtb r0, r1
    ] {
        let program = [insn, 0xE7FE];
        let mut reference = thumb_cpu(&program);
        let mut jitted = thumb_cpu(&program);
        for cpu in [&mut reference, &mut jitted] {
            cpu.state.set_reg(1, input);
        }
        reference.run(1).unwrap();
        jitted.run_cached(1).unwrap();
        assert!(
            jitted.compiled_block_count() > 0,
            "{insn:#06X} should be JIT compiled"
        );
        assert_eq!(digest(&reference), digest(&jitted), "{insn:#06X}");
    }
}

/// A conditional instruction after `subs` reads the flags that instruction produced. The app's dialog
/// scanner is exactly this shape (`sub ip, r2, #0xff00` / `subs ip, ip, #0xff` / `addne` / `strne`),
/// and a JIT that loses Z there would count zero objects and skip every drawing that follows.
#[test]
fn conditional_execution_after_subs_matches_jit() {
    let program = [
        0xE242CCFF,  // sub  ip, r2, #0xff00
        0xE25CC0FF,  // subs ip, ip, #0xff
        0x12877001,  // addne r7, r7, #1
        0xE1A08007,  // mov  r8, r7
        0xEAFF_FFFE, // b .
    ];
    let run = |jit: bool| {
        let mut cpu = new_cpu(&program);
        cpu.state.set_reg(2, 0x6003);
        cpu.state.set_reg(7, 0);
        cpu.set_jit_enabled(jit);
        if jit {
            cpu.run_cached(5).unwrap();
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be JIT compiled"
            );
        } else {
            cpu.run(5).unwrap();
        }
        (cpu.state.r, cpu.state.cpsr.bits())
    };
    assert_eq!(run(false), run(true), "condition after subs must agree");
}

/// The shape the dialog scanner actually runs: a conditional store followed by another conditional
/// instruction reading the *same* Z. A memory access is the instruction most likely to destroy the
/// host's flag register, so the second guard must still see the flags `subs` produced.
#[test]
fn conditional_store_then_conditional_add_matches_jit() {
    let program = [
        0xE242CCFF,  // sub   ip, r2, #0xff00
        0xE25CC0FF,  // subs  ip, ip, #0xff
        0x11A02007,  // movne r2, r7
        0x12883020,  // addne r3, r8, #0x20
        0x17830102,  // strne r0, [r3, r2, lsl #2]
        0x12877001,  // addne r7, r7, #1
        0xE1A09007,  // mov   sb, r7
        0xEAFF_FFFE, // b .
    ];
    let run = |jit: bool| {
        let mut cpu = new_cpu(&program);
        cpu.state.set_reg(2, 0x6003);
        cpu.state.set_reg(7, 0);
        cpu.state.set_reg(0, 0x1234);
        cpu.state.set_reg(8, DATA);
        cpu.set_jit_enabled(jit);
        if jit {
            cpu.run_cached(7).unwrap();
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be JIT compiled"
            );
        } else {
            cpu.run(7).unwrap();
        }
        digest(&cpu)
    };
    assert_eq!(
        run(false),
        run(true),
        "condition after a conditional store must agree"
    );
}

/// A guard is evaluated where it stands. A later flag write in the same block (`cmp` here) must not
/// retroactively decide an earlier `movne`: the dialog scanner has exactly this shape, and a guard
/// answered with the final flags counts no objects at all.
#[test]
fn a_guard_before_a_later_cmp_uses_the_earlier_flags() {
    let program = [
        0xE244CCFF,  // sub   ip, r4, #0xff00
        0xE25CC0FF,  // subs  ip, ip, #0xff   -> Z = 0
        0x13A03001,  // movne r3, #1
        0xE1520001,  // cmp   r2, r1          -> Z = 1
        0xBAFFFFFC,  // blt   back            (not taken: equal)
        0xE1A09003,  // mov   sb, r3
        0xEAFF_FFFE, // b .
    ];
    let run = |jit: bool| {
        let mut cpu = new_cpu(&program);
        cpu.state.set_reg(4, 0x6003);
        cpu.state.set_reg(1, 5);
        cpu.state.set_reg(2, 5);
        cpu.state.set_reg(3, 0);
        cpu.set_jit_enabled(jit);
        if jit {
            cpu.run_cached(6).unwrap();
            assert!(
                cpu.compiled_block_count() > 0,
                "the block should be JIT compiled"
            );
        } else {
            cpu.run(6).unwrap();
        }
        digest(&cpu)
    };
    assert_eq!(
        run(false),
        run(true),
        "an earlier guard must not read the later cmp's flags"
    );
}
