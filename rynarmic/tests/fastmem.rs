//! The boundary behaviour of inlined accesses (fastmem).
//!
//! The JIT inlines an access inside a window as one host `mov`: it takes it directly when `(addr - base) <= len - size`
//! and falls back to the host callback otherwise. **Behaviour outside a window has to match the callback byte for byte** (`SimpleMemory`'s
//! out-of-range read-as-0 and dropped writes), so every case here is checked by differencing the interpreter against the block cache
//! rather than asserting one value, which would stay green with both paths wrong.

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const MEMORY_SIZE: usize = 0x8000;
/// The window's last word.
const LAST_WORD: u32 = MEMORY_SIZE as u32 - 4;
/// The window's last byte.
const LAST_BYTE: u32 = MEMORY_SIZE as u32 - 1;
/// The first address out of range.
const PAST_END: u32 = MEMORY_SIZE as u32;

fn cpu_with(words: &[u32]) -> Cpu<SimpleMemory> {
    let mut memory = SimpleMemory::new(0, MEMORY_SIZE);
    memory.write_words(CODE, words);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu
}

/// A snapshot of one run: r0..r4, three memory words, and CPSR.
type Snapshot = ([u32; 5], [u32; 3], u32);

/// Runs all three paths (single-step, JIT with inlining, JIT with callbacks only) once each and compares them.
fn both_paths(
    words: &[u32],
    setup: impl Fn(&mut Cpu<SimpleMemory>),
    steps: u32,
) -> (Snapshot, Snapshot) {
    let mut results = Vec::new();
    for (jit, fastmem) in [(false, false), (true, true), (true, false)] {
        let mut cpu = cpu_with(words);
        cpu.set_jit_enabled(jit);
        cpu.set_fastmem_enabled(fastmem);
        setup(&mut cpu);
        if jit {
            cpu.run_cached(steps as u64).unwrap();
            assert!(
                cpu.compiled_block_count() > 0,
                "the jit path must really compile to machine code"
            );
        } else {
            cpu.run(steps as u64).unwrap();
        }
        results.push((
            [
                cpu.state.reg(0),
                cpu.state.reg(1),
                cpu.state.reg(2),
                cpu.state.reg(3),
                cpu.state.reg(4),
            ],
            [
                cpu.memory.peek_u32(LAST_WORD - 4),
                cpu.memory.peek_u32(LAST_WORD),
                cpu.memory.peek_u32(CODE),
            ],
            cpu.state.cpsr.bits(),
        ));
    }
    // Single-step, JIT+inlined and JIT callback-only all have to agree
    assert_eq!(
        results[0], results[1],
        "inlined accesses disagree with the interpreter"
    );
    assert_eq!(
        results[0], results[2],
        "inlining disagrees with the callback path"
    );
    (results[0], results[1])
}

#[test]
fn load_at_window_boundary_matches_interpreter() {
    let program = [
        0xE591_0000, // ldr r0, [r1]
        0xE1A0_0000, // mov r0, r0
        0xEAFF_FFFE, // b .
    ];
    for (addr, expect) in [
        (LAST_WORD, 0x1234_5678u32),
        (0x0000_0004, 0xAAAA_AAAA),
        (LAST_WORD - 4, 0),
        (LAST_WORD + 1, 0), // one byte past the window's end
        (LAST_BYTE, 0),
        (PAST_END, 0),
        (PAST_END + 0x1000, 0),
        (0xFFFF_FFFC, 0),
    ] {
        let (stepped, _) = both_paths(
            &program,
            |cpu| {
                cpu.state.set_reg(0, 0xDEAD_BEEF); // distinguishes "read 0" from "did not execute"
                cpu.state.set_reg(1, addr);
                cpu.memory.write_u32(LAST_WORD, 0x1234_5678);
                cpu.memory.write_u32(0x0000_0004, 0xAAAA_AAAA);
            },
            1,
        );
        assert_eq!(stepped.0[0], expect, "addr={addr:#x}");
    }
}

#[test]
fn byte_and_halfword_loads_at_boundary() {
    let program = [
        0xE5D1_0000, // ldrb r0, [r1]
        0xE1D1_20B0, // ldrh r2, [r1]
        0xEAFF_FFFE, // b .
    ];
    for (addr, expect_byte, expect_half) in [
        (LAST_BYTE, 0x5Au32, 0),
        (LAST_BYTE + 1, 0, 0),
        (LAST_WORD, 0x78, 0x5678),
        (LAST_WORD + 2, 0x34, 0x5A34), // setup rewrote the last byte as 0x5A
        (LAST_WORD + 3, 0x5A, 0),
        (PAST_END, 0, 0),
    ] {
        let (stepped, _) = both_paths(
            &program,
            |cpu| {
                cpu.state.set_reg(1, addr);
                cpu.memory.write_u32(LAST_WORD, 0x1234_5678);
                cpu.memory.write_u8(LAST_BYTE, 0x5A);
            },
            2,
        );
        assert_eq!(stepped.0[0], expect_byte, "ldrb addr={addr:#x}");
        assert_eq!(stepped.0[2], expect_half, "ldrh addr={addr:#x}");
    }
}

#[test]
fn store_beyond_window_is_dropped_like_the_callback() {
    let program = [
        0xE581_0000, // str r0, [r1]
        0xEAFF_FFFE, // b .
    ];
    for addr in [
        LAST_WORD,
        LAST_WORD + 1,
        LAST_WORD + 2,
        PAST_END,
        0xFFFF_FFFC,
    ] {
        let (stepped, _) = both_paths(
            &program,
            |cpu| {
                cpu.state.set_reg(0, 0xCAFE_BABE);
                cpu.state.set_reg(1, addr);
                cpu.memory.write_u32(LAST_WORD, 0x1111_1111);
                cpu.memory.write_u32(LAST_WORD - 4, 0x2222_2222);
            },
            1,
        );
        let expect = if addr == LAST_WORD {
            0xCAFE_BABE
        } else {
            0x1111_1111
        };
        assert_eq!(stepped.1[1], expect, "str addr={addr:#x}");
        assert_eq!(
            stepped.1[0], 0x2222_2222,
            "the neighbouring word must not be damaged"
        );
    }
}

#[test]
fn ldm_mixes_inlined_and_callback_words() {
    // A two-word batch read crosses the window's end: the first word is inlined and the second must fall back to the callback and read 0
    let program = [
        0xE891_000C, // ldmia r1, {r2, r3}
        0xEAFF_FFFE, // b .
    ];
    let (stepped, _) = both_paths(
        &program,
        |cpu| {
            cpu.state.set_reg(1, LAST_WORD);
            cpu.state.set_reg(2, 0xDEAD_BEEF);
            cpu.state.set_reg(3, 0xDEAD_BEEF);
            cpu.memory.write_u32(LAST_WORD, 0x0BAD_F00D);
        },
        1,
    );
    assert_eq!(stepped.0[0], 0, "r0 should be untouched");
    assert_eq!(
        stepped.0[2], 0x0BAD_F00D,
        "the word inside the window must be read"
    );
    assert_eq!(
        stepped.0[3], 0,
        "the word out of range reads 0, as the callback does"
    );
    assert!(
        stepped.1[1] != 0,
        "the last word should keep its value (this is a read, not a write)"
    );
}

/// Turning inlining off only swaps the implementation; the semantics may not move.
#[test]
fn fastmem_toggle_is_semantically_neutral() {
    let program = [
        0xE591_0000, // ldr r0, [r1]
        0xE581_0004, // str r0, [r1, #4]
        0xE591_2004, // ldr r2, [r1, #4]
        0xEAFF_FFFE, // b .
    ];
    let (stepped, _) = both_paths(
        &program,
        |cpu| {
            cpu.state.set_reg(1, LAST_WORD - 4);
            cpu.memory.write_u32(LAST_WORD - 4, 0xFEED_FACE);
        },
        3,
    );
    assert_eq!(stepped.0[0], 0xFEED_FACE);
    assert_eq!(stepped.0[2], 0xFEED_FACE);
}

/// Indexed writeback: post-index +4, pre-index +4 with writeback, a negative pre-index with writeback, and a negative offset without.
///
/// These cover the four arms of `emit_address`'s immediate-offset path (also the shape `memcpy` produces),
/// where a wrong writeback value shows up immediately in r1 or r2.
#[test]
fn indexed_addressing_writes_back_correctly() {
    let program = [
        0xE491_0004, // ldr r0, [r1], #4      post-index: access at r1, write back r1+4
        0xE5B1_3004, // ldr r3, [r1, #4]!     pre-index: access r1+4, write back r1+4
        0xE522_0004, // str r0, [r2, #-4]!    negative pre-index
        0xE511_4008, // ldr r4, [r1, #-8]     negative offset, no writeback
        0xEAFF_FFFE, // b .
    ];
    for base in [0x100u32, LAST_WORD - 0x40] {
        let (stepped, _) = both_paths(
            &program,
            |cpu| {
                cpu.state.set_reg(1, base);
                cpu.state.set_reg(2, base + 0x20);
                cpu.memory.write_u32(base, 0x1111_1111);
                cpu.memory.write_u32(base + 4, 0x2222_2222);
                cpu.memory.write_u32(base + 8, 0x3333_3333);
                cpu.memory.write_u32(base + 0x1C, 0x4444_4444);
            },
            4,
        );
        // Post-index reads base; then r1 = base+4, and the pre-index reads base+8 and writes back base+8
        assert_eq!(stepped.0[0], 0x1111_1111, "base={base:#x}");
        assert_eq!(stepped.0[3], 0x3333_3333, "base={base:#x}");
        assert_eq!(
            stepped.0[1],
            base + 8,
            "wrong writeback value: base={base:#x}"
        );
        // A negative offset without writeback: reads base+8-8 = base
        assert_eq!(stepped.0[4], 0x1111_1111, "base={base:#x}");
        // r2 with pre-index writeback: str r0, [r2, #-4]! writes r2-4 and then writes back r2-4
        assert_eq!(
            stepped.0[2] as i64,
            base as i64 + 0x20 - 4,
            "base={base:#x}"
        );
    }
}
