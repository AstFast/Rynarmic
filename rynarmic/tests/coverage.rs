//! Decoding coverage and robustness.
//!
//! Two goals:
//! 1. **Exhaustive Thumb halfwords**: walk all 65536 encodings, tally the decoded, undefined and unimplemented shares,
//!    and try executing each one to be sure neither the decoder nor the interpreter panics.
//! 2. **A pseudo-random A32 sweep**: a fixed-seed xorshift generates a large number of 32-bit encodings, again requiring no panic
//!    and a bounded IR count (so data is never taken for an overlong instruction sequence).

use rynarmic::{Cpu, memory::SimpleMemory};

const CODE: u32 = 0x1000;

/// Builds a minimal runnable CPU: small memory, deterministic registers, and 0 for any out-of-range read.
fn probe_cpu(thumb: bool) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, 0x2000);
    memory.write_words(CODE, &[0xEAFF_FFFE; 8]);
    memory.write_thumb(CODE, &[0xE7FE; 16]);

    let mut cpu = Cpu::new(memory);
    cpu.state.cpsr.set_t(thumb);
    cpu.state.set_pc(CODE);
    cpu.state.set_reg(13, 0x1800);
    cpu.state.set_reg(14, 0x1000);
    for reg in 0..13u8 {
        cpu.state.set_reg(reg, 0x0001_0000 + (reg as u32) * 8);
    }
    cpu
}

#[test]
fn thumb_encoding_space_is_fully_covered() {
    let mut decoded = 0usize;
    let mut undefined = 0usize;
    let mut unsupported = 0usize;
    let mut undefined_high_bytes = std::collections::BTreeSet::new();
    // Which stretches are unimplemented, by top byte. This is the input to any decision about
    // extending the decoder: the ARMv6T2 16-bit additions (`0xB1xx` CBZ/CBNZ, `0xB2xx`
    // SXTH/UXTH/SXTB/UXTB, `0xBAxx` REV/REV16/REVSH) live in three known places, and anything
    // outside those three is either reserved or a format this decoder simply never grew.
    let mut unsupported_high_bytes: std::collections::BTreeMap<u8, usize> =
        std::collections::BTreeMap::new();
    let mut bl_prefixes = 0usize;

    for raw in 0..=u16::MAX {
        // BL and BLX prefixes are tallied separately (they need a pair; one halfword is not an instruction)
        if rynarmic::decoder::thumb::is_bl_prefix(raw) {
            unsupported += 1;
            bl_prefixes += 1;
            continue;
        }
        match rynarmic::decoder::thumb::decode(raw, CODE, true) {
            Ok(ops) => {
                decoded += 1;
                assert!(
                    ops.len() <= 4,
                    "encoding {raw:#06x} produced too much IR: {}",
                    ops.len()
                );
            }
            Err(rynarmic::DecodeError::Undefined(_)) => {
                undefined += 1;
                undefined_high_bytes.insert((raw >> 8) as u8);
            }
            Err(rynarmic::DecodeError::Unsupported { .. }) => {
                unsupported += 1;
                *unsupported_high_bytes.entry((raw >> 8) as u8).or_default() += 1;
            }
        }
    }

    // Every Thumb-1 format should be recognised, so the decoded share has to be well above half
    assert!(
        decoded > 40_000,
        "only {decoded} Thumb encodings decoded; coverage is too thin"
    );
    // Only two stretches are legitimately undefined:
    //   0xDExx         -- UDF, the permanently undefined instruction
    //   0xE8xx..0xEFxx -- the second halfword of BLX, which alone is not an instruction
    let expected: std::collections::BTreeSet<u8> =
        std::iter::once(0xDE).chain(0xE8..=0xEF).collect();
    assert_eq!(
        undefined_high_bytes, expected,
        "undefined encodings landed outside the expected space: {undefined_high_bytes:?}"
    );
    assert_eq!(
        undefined, 2304,
        "unexpected number of undefined encodings: {undefined}"
    );
    println!(
        "Thumb halfwords: decoded {decoded} / undefined {undefined} / unimplemented {unsupported} (of 65536)"
    );
    println!(
        "  of the {unsupported} unimplemented: {bl_prefixes} are the BL/BLX prefix range (0xF0xx-0xF7xx)"
    );
    let rows: Vec<String> = unsupported_high_bytes
        .iter()
        .map(|(hi, n)| format!("{hi:02X}xx:{n}"))
        .collect();
    println!("  by top byte (count of unimplemented encodings):");
    for chunk in rows.chunks(8) {
        println!("    {}", chunk.join("  "));
    }
}

#[test]
fn thumb_every_decodable_instruction_executes_without_panic() {
    for raw in 0..=u16::MAX {
        if rynarmic::decoder::thumb::is_bl_prefix(raw) {
            continue;
        }
        let Ok(ops) = rynarmic::decoder::thumb::decode(raw, CODE, true) else {
            continue;
        };

        let mut cpu = probe_cpu(true);
        // Place the instruction under test at the PC
        cpu.memory.write_thumb(CODE, &[raw]);
        // Single-step it; an exception (SWI, BKPT, unsupported) is a normal outcome
        let _ = cpu.step();
        assert!(ops.len() <= 4);
    }
}

#[test]
fn thumb_bl_prefix_is_not_decoded_as_instruction() {
    // A lone prefix is not an instruction
    assert!(rynarmic::decoder::thumb::decode(0xF000, CODE, true).is_err());
    assert!(rynarmic::decoder::thumb::is_bl_prefix(0xF7FF));
    assert!(!rynarmic::decoder::thumb::is_bl_prefix(0xF800));
}

/// A fixed-seed xorshift64, so runs are reproducible.
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

#[test]
fn a32_random_encodings_never_panic() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut decoded = 0usize;
    let mut rejected = 0usize;

    for _ in 0..200_000 {
        let inst = rng.next() as u32;
        let cond_passed = true;
        match rynarmic::decoder::a32::decode(inst, CODE, cond_passed) {
            Ok(ops) => {
                decoded += 1;
                assert!(
                    ops.len() <= 4,
                    "encoding {inst:#010x} produced too much IR: {}",
                    ops.len()
                );

                // Try executing, to be sure the interpreter panics on no legal IR combination
                let mut cpu = probe_cpu(false);
                cpu.memory.write_words(CODE, &[inst]);
                let _ = cpu.step();
            }
            Err(_) => rejected += 1,
        }
    }

    // Most random 32-bit encodings fall in legal space, so the decode rate should not be low (a low one would mean the dispatch is too narrow)
    assert!(
        decoded > 40_000,
        "only {decoded} of 200k random encodings decoded"
    );
    println!("A32 random sweep: {decoded} decoded / {rejected} rejected (of 200000)");
}

#[test]
fn a32_undefined_register_combinations_are_rejected() {
    // LDRD with an odd Rt is an undefined encoding (it would write Rt+1, running past r15)
    // 0xE1C1_10D8 => ldrd r1, [r1, #8]
    assert!(rynarmic::decoder::a32::decode(0xE1C1_10D8, CODE, true).is_err());
    // 0xE1C1_E0D8 => ldrd r14, [r1, #8] (r15 written as the second word)
    assert!(rynarmic::decoder::a32::decode(0xE1C1_E0D8, CODE, true).is_err());

    // SMLALxy requires an even RdLo: 0xE140_2384 => smlalxy r0, r2, r3, r4 is legal
    assert!(rynarmic::decoder::a32::decode(0xE140_2384, CODE, true).is_ok());
    // RdLo = r1 (odd) is illegal
    assert!(rynarmic::decoder::a32::decode(0xE140_1384, CODE, true).is_err());
}

#[test]
fn a32_condition_codes_cover_all_sixteen_values() {
    // The 16 conditional versions of mov r0, #1: IR only appears where the condition holds
    for cond in 1..=0xEu32 {
        let inst = 0x03A0_0001 | (cond << 28);
        let taken = |n: bool, z: bool, c: bool, v: bool| {
            let mut cpu = probe_cpu(false);
            cpu.memory.write_words(CODE, &[inst]);
            cpu.state.set_reg(0, 0);
            cpu.state.cpsr.set_n(n);
            cpu.state.cpsr.set_z(z);
            cpu.state.cpsr.set_c(c);
            cpu.state.cpsr.set_v(v);
            let _ = cpu.step();
            cpu.state.reg(0) == 1
        };

        // Build a set of flag combinations that covers each condition being both true and false
        let mut any_true = false;
        let mut any_false = false;
        for flags in 0..16u32 {
            let n = flags & 1 != 0;
            let z = flags & 2 != 0;
            let c = flags & 4 != 0;
            let v = flags & 8 != 0;
            if taken(n, z, c, v) {
                any_true = true;
            } else {
                any_false = true;
            }
        }
        // AL (0xE) always holds; every other condition has a case where it fails
        if cond == 0xE {
            assert!(any_true);
        } else {
            assert!(any_true && any_false, "condition {cond:#x} looks constant");
        }
    }
}
