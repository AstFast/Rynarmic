//! Hand-assembled kernels with realistic shapes.
//!
//! They serve as the baseline for optimizations that need realistic code shapes (fastmem and onwards): branch-dense, longer blocks than a synthetic loop,
//! and a mix of accesses and arithmetic. A synthetic benchmark (`examples/bench.rs`) cannot show what real code gains,
//! because its loop bodies are too clean and its blocks too small.
//!
//! **Why hand-assembled**: no ARM cross toolchain is available here, and `armv5te-none-eabi` is a Tier 3 target
//! (needing nightly and build-std). So these are a few small functions assembled bit by bit, with the mnemonics written out in comments.
//! A wrong encoding does not pass silently: `tests/real_workload.rs` asserts on **expected values** (quotient, remainder, copy result)
//! rather than on "the two paths agree", which stays green when both are wrong.
//!
//! The limitation, recorded in the plan's known limits: this is not CoreMark, only a shape **close** to real compiler output.
//! Until real firmware or CoreMark is wired in, any "real code runs at x MIPS" claim is a lower bound.

pub const CODE: u32 = 0x1000;
pub const SRC: u32 = 0x2000;
pub const DST: u32 = 0x3000;

/// The A32 data-processing opcodes (bits 24-21).
mod op {
    pub const SUB: u32 = 0b0010;
    pub const ADD: u32 = 0b0100;
    pub const CMP: u32 = 0b1010;
    pub const ORR: u32 = 0b1100;
    pub const MOV: u32 = 0b1101;
}

/// The shift types (bits 6-5).
mod shift {
    pub const LSL: u32 = 0b00;
    pub const LSR: u32 = 0b01;
}

/// The condition codes.
pub mod cond {
    pub const NE: u32 = 0x1;
    pub const CS: u32 = 0x2;
    pub const AL: u32 = 0xE;
}

/// Register-shifted data processing: `<op>{s}{cond} rd, rn, rm <shift> #amount`
///
/// The parameters are the instruction's fields; a struct would only obscure how they line up.
#[allow(clippy::too_many_arguments)]
fn dp_reg(cc: u32, opcode: u32, s: bool, rn: u8, rd: u8, rm: u8, stype: u32, amount: u32) -> u32 {
    (cc << 28)
        | (opcode << 21)
        | ((s as u32) << 20)
        | ((rn as u32) << 16)
        | ((rd as u32) << 12)
        | ((amount & 0x1F) << 7)
        | (stype << 5)
        | rm as u32
}

/// Immediate data processing (rotate = 0, so the immediate has to be under 256).
fn dp_imm(cc: u32, opcode: u32, s: bool, rn: u8, rd: u8, imm: u32) -> u32 {
    assert!(
        imm < 256,
        "only 8-bit immediates with rotate = 0 are used here"
    );
    (cc << 28)
        | (1 << 25)
        | (opcode << 21)
        | ((s as u32) << 20)
        | ((rn as u32) << 16)
        | ((rd as u32) << 12)
        | imm
}

fn mov_imm(rd: u8, imm: u32) -> u32 {
    dp_imm(cond::AL, op::MOV, false, 0, rd, imm)
}

fn cmp_reg(rn: u8, rm: u8) -> u32 {
    dp_reg(cond::AL, op::CMP, true, rn, 0, rm, shift::LSL, 0)
}

fn subs_imm(rd: u8, rn: u8, imm: u32) -> u32 {
    dp_imm(cond::AL, op::SUB, true, rn, rd, imm)
}

/// `b{cond} target`, where `from` is this instruction's own address.
fn branch(cc: u32, from: u32, target: u32) -> u32 {
    let offset = target.wrapping_sub(from.wrapping_add(8));
    assert_eq!(offset & 3, 0, "a branch target has to be 4-byte aligned");
    (cc << 28) | 0x0A00_0000 | ((offset >> 2) & 0x00FF_FFFF)
}

/// `bl target`
fn bl(from: u32, target: u32) -> u32 {
    branch(cond::AL, from, target) | (1 << 24)
}

/// `bx rm`
fn bx(rm: u8) -> u32 {
    0xE12F_FF10 | rm as u32
}

/// `ldr rd, [rn], #imm`
fn ldr_post(rd: u8, rn: u8, imm: u32) -> u32 {
    0xE490_0000 | ((rn as u32) << 16) | ((rd as u32) << 12) | (imm & 0xFFF)
}

/// `str rd, [rn], #imm`
fn str_post(rd: u8, rn: u8, imm: u32) -> u32 {
    0xE480_0000 | ((rn as u32) << 16) | ((rd as u32) << 12) | (imm & 0xFFF)
}

/// Shift-and-subtract division, shaped like libgcc's `__aeabi_uidiv`:
/// on entry r0 is the dividend and r1 the divisor; on exit r0 holds the quotient and r2 the remainder (r0 is clobbered).
///
/// 32 rounds of nine instructions, with four conditional or branch instructions per block — a shape no synthetic benchmark has.
pub fn udiv_program() -> Vec<u32> {
    const INIT: u32 = 3;
    const PER_ROUND: u32 = 9;
    let loop_head = CODE + INIT * 4;
    let bne_pc = loop_head + (PER_ROUND - 1) * 4;

    vec![
        mov_imm(2, 0),  // mov  r2, #0          ; the remainder
        mov_imm(3, 0),  // mov  r3, #0          ; the quotient
        mov_imm(4, 32), // mov  r4, #32         ; 32 bits to go
        dp_reg(cond::AL, op::MOV, false, 0, 5, 0, shift::LSR, 31), // mov  r5, r0, lsr #31
        dp_reg(cond::AL, op::ORR, false, 5, 2, 2, shift::LSL, 1), // orr  r2, r5, r2, lsl #1
        dp_reg(cond::AL, op::MOV, false, 0, 0, 0, shift::LSL, 1), // mov  r0, r0, lsl #1
        dp_reg(cond::AL, op::MOV, false, 0, 3, 3, shift::LSL, 1), // mov  r3, r3, lsl #1
        cmp_reg(2, 1),  // cmp  r2, r1
        dp_reg(cond::CS, op::SUB, false, 2, 2, 1, shift::LSL, 0), // subcs r2, r2, r1
        dp_imm(cond::CS, op::ADD, false, 3, 3, 1), // addcs r3, r3, #1
        subs_imm(4, 4, 1), // subs r4, r4, #1
        branch(cond::NE, bne_pc, loop_head), // bne  loop
        dp_reg(cond::AL, op::MOV, false, 0, 0, 3, shift::LSL, 0), // mov r0, r3
        branch(cond::AL, CODE + 0x34, CODE + 0x34), // b .   (spins in place)
    ]
}

/// The division kernel's instruction count (the correctness case advances by exactly this, not by guessing).
pub const UDIV_INSTRUCTIONS: u64 = 3 + 32 * 9 + 1;

/// An unrolled-by-four word copy: four reads and four writes per iteration, ten instructions per block.
/// On entry r0 is the destination, r1 the source and r2 the word count (a multiple of four).
pub fn memcpy_program() -> Vec<u32> {
    const PER_ROUND: u32 = 10;
    let bne_pc = CODE + (PER_ROUND - 1) * 4;

    vec![
        ldr_post(3, 1, 4),                          // ldr r3, [r1], #4
        ldr_post(4, 1, 4),                          // ldr r4, [r1], #4
        ldr_post(5, 1, 4),                          // ldr r5, [r1], #4
        ldr_post(6, 1, 4),                          // ldr r6, [r1], #4
        str_post(3, 0, 4),                          // str r3, [r0], #4
        str_post(4, 0, 4),                          // str r4, [r0], #4
        str_post(5, 0, 4),                          // str r5, [r0], #4
        str_post(6, 0, 4),                          // str r6, [r0], #4
        subs_imm(2, 2, 4),                          // subs r2, r2, #4
        branch(cond::NE, bne_pc, CODE),             // bne  loop
        branch(cond::AL, CODE + 0x28, CODE + 0x28), // b .
    ]
}

/// A call-and-return shape: the loop `bl`s a leaf function that does `add; bx lr`.
/// On exit r5 holds the accumulator and r4 is 0.
pub fn call_program() -> Vec<u32> {
    const INIT: u32 = 2;
    const PER_ROUND: u32 = 5; // mov / bl / add / subs / bne
    let loop_head = CODE + INIT * 4;
    let bne_pc = loop_head + (PER_ROUND - 1) * 4;
    let func = CODE + 0x40;

    vec![
        mov_imm(4, 200), // mov r4, #200        ; the iteration count
        mov_imm(5, 0),   // mov r5, #0          ; the accumulator
        dp_reg(cond::AL, op::MOV, false, 0, 0, 4, shift::LSL, 0), // mov r0, r4
        bl(CODE + 12, func), // bl  func
        dp_reg(cond::AL, op::ADD, false, 5, 5, 0, shift::LSL, 0), // add r5, r5, r0
        subs_imm(4, 4, 1), // subs r4, r4, #1
        branch(cond::NE, bne_pc, loop_head), // bne loop
        branch(cond::AL, CODE + 0x1C, CODE + 0x1C), // b .   (spins in place)
    ]
}

/// The call kernel's leaf function, patched in: its body goes at `CODE + 0x40` and the caller writes it into the same memory.
pub fn call_program_with_function() -> Vec<u32> {
    let mut program = call_program();
    program.resize(0x40 / 4, 0); // pad up to the function entry with 0 (all andeq, never reached)
    // func: add r0, r0, #3 ; bx lr
    program.push(dp_imm(cond::AL, op::ADD, false, 0, 0, 3));
    program.push(bx(14));
    program
}

/// The instruction count of `call_program`: two initialisers, 200 rounds of five loop instructions plus two in the callee, and one `b .`.
pub const CALL_INSTRUCTIONS: u64 = 2 + 200 * 7 + 1;
