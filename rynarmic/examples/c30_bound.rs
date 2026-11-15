//! Experiment: the upper bound on what register caching can buy.
//!
//! (1) Compile the sixteen-instruction hot loop of `__udivsi3` into a block and count the state-struct memory operations
//!     (`[rbx+disp32]`) in the machine code and their share — those are what register caching would remove.
//! (2) Measure, with a host-side microbenchmark, what a memory round trip per register costs a serial dependency chain.
//!
//! Usage: `cargo run --release --example c30_bound`

use rynarmic::{Memory, backend, block, memory::SimpleMemory};

const CODE: u32 = 0x1000;

/// The hot loop of `__udivsi3` (0x1168–0x11a4).
const UDIV_LOOP: [u32; 16] = [
    0xE150_0001, // cmp   r0, r1
    0x2040_0001, // subcs r0, r0, r1
    0x2182_2003, // orrcs r2, r2, r3
    0xE150_00A1, // cmp   r0, r1, lsr #1
    0x2040_00A1, // subcs r0, r0, r1, lsr #1
    0x2182_20A3, // orrcs r2, r2, r3, lsr #1
    0xE150_0121, // cmp   r0, r1, lsr #2
    0x2040_0121, // subcs r0, r0, r1, lsr #2
    0x2182_2123, // orrcs r2, r2, r3, lsr #2
    0xE150_01A1, // cmp   r0, r1, lsr #3
    0x2040_01A1, // subcs r0, r0, r1, lsr #3
    0x2182_21A3, // orrcs r2, r2, r3, lsr #3
    0xE350_0000, // cmp   r0, #0
    0x11B0_3223, // lsrsne r3, r3, #4
    0x11A0_1221, // lsrne r1, r1, #4
    0x1AFF_FFEF, // bne 0x1168
];

/// A copy of the kernel's loop body (the same shape as tests/support/kernels.rs, simplified to the unrolled-free form).
const COPY_LOOP: [u32; 10] = [
    0xE590_2000, // ldr r2, [r0]
    0xE5D2_1000, // ldrb r1, [r2]     -- two accesses stand in for one copy iteration
    0xE5C1_0000, // strb r1, [r0]
    0xE280_0004, // add r0, r0, #4
    0xE150_0005, // cmp r0, r5
    0x3AFF_FFFA, // bls 0x1000
    0xE1A0_0000, // mov r0, r0
    0xE1A0_0000,
    0xE1A0_0000,
    0xEAFF_FFFD, // b 0x1004 (approximate)
];

fn compile(words: &[u32]) -> (Vec<u8>, u32) {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(CODE, words);
    let translated = match block::translate(&mut memory, CODE, false).unwrap() {
        block::Translation::Block(b) => b,
        _ => panic!("cannot form a block"),
    };
    let count = translated.instruction_count;
    let window = memory.fast_window();
    let mut windows = [None; rynarmic::memory::MAX_WINDOWS];
    windows[0] = window;
    let mut arena = backend::CodeArena::new(1 << 20);
    let jit = backend::compile::<SimpleMemory>(&translated, false, windows, &mut arena).unwrap();
    let ptr = jit.entry_ptr();
    let bytes = (0..jit.code_len())
        .map(|i| unsafe { *ptr.add(i) })
        .collect();
    (bytes, count)
}

/// Counts accesses of the form `[rbx+disp32]` (ModRM mod=10, rm=SIB, with SIB base=rbx).
/// This backend always uses SIB encoding, so `ModRM in {0x84,0x8C,0x94,0x9C,0xA4,0xAC,0xB4,0xBC} + SIB 0x23`
/// means an access to the state struct (or to a callback argument derived from it).
/// A classified count of state-struct accesses (`[rbx+disp32]`).
///
/// A REX prefix may sit one or two bytes before the ModRM, so at each SIB-form ModRM the real
/// opcode is found by looking backwards: `mov` (8B/89/8A/88) is a **pure move**, which register caching removes outright;
/// anything else with a memory operand (add/sub/or/and/cmp/movzx and so on) is **fused**: the instruction stays,
/// and only the L1 access and the store-forwarding wait are saved.
fn classify_state_ops(bytes: &[u8]) -> (usize, usize) {
    let mut pure = 0;
    let mut fused = 0;
    for i in 1..bytes.len().saturating_sub(1) {
        let (modrm, sib) = (bytes[i], bytes[i + 1]);
        if !(0x84..=0xBF).contains(&modrm) || modrm & 0xC7 != 0x84 || sib & 0x3F != 0x23 {
            continue;
        }
        // Look backwards for the opcode, skipping REX 40-4F
        let mut j = i;
        while j > 0 && (0x40..=0x4F).contains(&bytes[j - 1]) {
            j -= 1;
        }
        if j == 0 {
            continue;
        }
        match bytes[j - 1] {
            0x88..=0x8B => pure += 1,
            _ => fused += 1,
        }
    }
    (pure, fused)
}

fn report(name: &str, words: &[u32]) {
    let (bytes, count) = compile(words);
    let (pure, fused) = classify_state_ops(&bytes);
    let state_ops = pure + fused;
    // Discount the fixed prologue, epilogue and link slot (a single-instruction `b .` block measures 123 bytes)
    let fixed = 123usize;
    let body = bytes.len().saturating_sub(fixed);
    println!(
        "{name:12} {count:3} guest insns  block {:4} bytes (body ~{body:4})  state accesses {state_ops:3} (pure moves {pure}, fused {fused})  ~ {:.1} per insn",
        bytes.len(),
        state_ops as f64 / count as f64
    );
}

fn main() {
    report("udiv loop", &UDIV_LOOP);
    report("copy shape", &COPY_LOOP);

    // (2) Microbenchmark: the same read-modify-write serial chain, once through memory (what the JIT emits today)
    //    and once resident in a register (the goal). The difference is the per-instruction round trip cost.
    const N: u64 = 200_000_000;
    // An opaque pointer stops LLVM from proving there is no aliasing, forcing the same shape the JIT produces:
    // every round really loads r0, adds and stores it, and the next round's load depends on the previous store
    // (x86's store-to-load forwarding, exactly the serial cost every guest instruction pays today).
    let mut state = [0u32; 16];
    let p0 = std::hint::black_box(&mut state[0]) as *mut u32;
    let p1 = std::hint::black_box(&mut state[1]) as *mut u32;
    let start = std::time::Instant::now();
    for _ in 0..N {
        unsafe { *p0 = (*p0).wrapping_add(*p1) };
    }
    let mem_ns = start.elapsed().as_secs_f64() * 1e9 / N as f64;
    std::hint::black_box(&state);

    let (mut a, b) = (0u32, 1u32);
    let start = std::time::Instant::now();
    for _ in 0..N {
        a = a.wrapping_add(std::hint::black_box(b));
    }
    let reg_ns = start.elapsed().as_secs_f64() * 1e9 / N as f64;
    std::hint::black_box((a, b));

    println!();
    println!(
        "read-modify-write chain: through memory {:.2} ns/op, register-resident {:.2} ns/op, difference {:.2} ns/op",
        mem_ns,
        reg_ns,
        mem_ns - reg_ns
    );
    println!(
        "  (at 4 GHz that is about {:.1} against {:.1} cycles; the difference is what register caching can take off a **serial chain**)",
        mem_ns * 4.0,
        reg_ns * 4.0
    );
}
