//! Bit operations, the barrel shifter, and condition evaluation: the pieces whose semantics come
//! straight from the ISA.

/// The barrel shifter's modes.
///
/// `Rrx` is produced only by A32's immediate `ROR #0`: rotate right by one, with the vacated top
/// bit taken from the carry flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
    Rrx,
}

/// Bit `index`, counted from zero.
#[inline]
pub const fn bit(value: u32, index: u32) -> u32 {
    (value >> index) & 1
}

/// The bitfield spanning the closed interval `[lo, hi]`.
#[inline]
pub fn bits(value: u32, hi: u32, lo: u32) -> u32 {
    debug_assert!(hi >= lo && hi < 32);
    let width = hi - lo + 1;
    let mask = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    (value >> lo) & mask
}

/// Sign-extends a `from_bits`-wide value to 32 bits.
#[inline]
pub fn sign_extend(value: u32, from_bits: u32) -> u32 {
    debug_assert!(from_bits > 0 && from_bits <= 32);
    if from_bits == 32 {
        return value;
    }
    let shift = 32 - from_bits;
    (((value << shift) as i32) >> shift) as u32
}

/// Rotate right. `amount` is taken modulo 32.
#[inline]
pub fn ror(value: u32, amount: u32) -> u32 {
    value.rotate_right(amount & 31)
}

/// ARM's 8-bit immediate expansion (`ARMExpandImm_C`).
///
/// Returns `(expanded value, whether the carry is affected)`. With `rotate == 0` the carry keeps
/// `carry_in`.
#[inline]
pub fn expand_imm_c(imm12: u32, carry_in: bool) -> (u32, bool) {
    let imm8 = imm12 & 0xFF;
    let rotate = (imm12 >> 8) & 0xF;
    if rotate == 0 {
        (imm8, carry_in)
    } else {
        let value = ror(imm8, rotate * 2);
        (value, (value >> 31) != 0)
    }
}

/// Add with carry. Returns `(result, carry out, overflow)`.
#[inline]
pub fn add_with_carry(x: u32, y: u32, carry_in: bool) -> (u32, bool, bool) {
    let (partial, carry_a) = x.overflowing_add(y);
    let (result, carry_b) = partial.overflowing_add(carry_in as u32);
    let carry_out = carry_a || carry_b;
    let overflow = ((x ^ result) & (y ^ result)) & 0x8000_0000 != 0;
    (result, carry_out, overflow)
}

/// Performs one shift and reports the carry it produces (`Shift_C`).
///
/// `amount` is the raw shift amount: LSL/LSR/ASR accept 0..=32, ROR accepts 0..=31 or 32. `Rrx` is
/// A32's `ROR #0`: the amount takes no part in the computation and the vacated bit comes from
/// `carry_in`.
#[inline]
pub fn shift_c(value: u32, shift: ShiftType, amount: u32, carry_in: bool) -> (u32, bool) {
    match shift {
        ShiftType::Lsl => match amount {
            0 => (value, carry_in),
            1..=31 => (value << amount, bit(value, 32 - amount) != 0),
            32 => (0, bit(value, 0) != 0),
            _ => (0, false),
        },
        ShiftType::Lsr => match amount {
            0 => (value, carry_in),
            1..=31 => (value >> amount, bit(value, amount - 1) != 0),
            32 => (0, bit(value, 31) != 0),
            _ => (0, false),
        },
        ShiftType::Asr => match amount {
            0 => (value, carry_in),
            1..=31 => (
                ((value as i32) >> amount) as u32,
                bit(value, amount - 1) != 0,
            ),
            _ => (((value as i32) >> 31) as u32, bit(value, 31) != 0),
        },
        ShiftType::Ror => match amount {
            0 => (value, carry_in),
            32 => (value, bit(value, 31) != 0),
            _ => {
                let value = ror(value, amount);
                (value, bit(value, 31) != 0)
            }
        },
        ShiftType::Rrx => (
            (u32::from(carry_in) << 31) | (value >> 1),
            bit(value, 0) != 0,
        ),
    }
}

/// Whether the condition code is satisfied (`ConditionPassed`).
#[inline]
pub fn cond_holds(cond: u32, n: bool, z: bool, c: bool, v: bool) -> bool {
    match cond {
        0x0 => z,
        0x1 => !z,
        0x2 => c,
        0x3 => !c,
        0x4 => n,
        0x5 => !n,
        0x6 => v,
        0x7 => !v,
        0x8 => c && !z,
        0x9 => !c || z,
        0xA => n == v,
        0xB => n != v,
        0xC => !z && n == v,
        0xD => z || n != v,
        0xE => true,
        _ => false, // 0xF is the unconditional space, which the decoder handles separately
    }
}
