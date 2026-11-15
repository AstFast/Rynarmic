//! An emission-quality audit: machine-code bytes per block, and the **marginal** cost with the fixed prologue and epilogue discounted.
//!
//! This measure does not depend on machine load, which makes it the steady basis for "how many host instructions does the backend emit
//! per guest instruction" and a far better A/B metric than MIPS (see the efficiency plan in `docs/PLAN.md`).
//!
//! The fill measurements use "a single-instruction `b .` block is 123 bytes" as the fixed-overhead baseline, with about ±2 bytes of noise;
//! for exact **instruction counts** use `examples/dump_code.rs` and read the disassembly, which is the only authoritative view.
//!
//! Usage: `cargo run --release --example code_stats`

use rynarmic::{Memory, backend, block, memory::SimpleMemory};

const CODE: u32 = 0x1000;
const TAIL: u32 = 0xEAFF_FFFE; // b .

fn block_bytes(words: &[u32]) -> Option<(usize, u32)> {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(CODE, words);
    let translated = match block::translate(&mut memory, CODE, false).ok()? {
        block::Translation::Block(block) => block,
        _ => return None,
    };
    let window = memory.fast_window();
    let mut windows = [None; rynarmic::memory::MAX_WINDOWS];
    windows[0] = window;
    let mut arena = backend::CodeArena::new(1 << 20);
    let jit = backend::compile::<SimpleMemory>(&translated, false, windows, &mut arena)?;
    Some((jit.code_len(), translated.instruction_count))
}

/// Absolute size: the whole block, prologue and epilogue included.
fn absolute(name: &str, words: &[u32]) {
    match block_bytes(words) {
        Some((bytes, count)) => println!(
            "{name:28} {count:3} insns {bytes:5} bytes  {:6.1} bytes/insn (fixed overhead included)",
            bytes as f64 / count as f64
        ),
        None => println!("{name:28} unavailable"),
    }
}

/// Per-instruction cost: fill 31 copies of one instruction and discount the fixed overhead (a `b .` block is 123 bytes).
///
/// Steadier than differencing two and five copies: the fixed overhead is **measured directly** rather than estimated.
fn fill(name: &str, instruction: u32) {
    let fixed = match block_bytes(&[TAIL]) {
        Some((bytes, _)) => bytes,
        None => return,
    };
    let mut words = vec![instruction; 31];
    words.push(TAIL);
    match block_bytes(&words) {
        Some((bytes, count)) if count >= 31 => {
            let per = (bytes.saturating_sub(fixed)) as f64 / 31.0;
            println!(
                "{name:28} {:6.2} bytes/insn  ~ {:4.1} host insns",
                per,
                per / 4.2
            );
        }
        _ => println!("{name:28} unavailable"),
    }
}

/// Marginal cost: repeat a stretch two and five times, and divide the difference by the instruction count.
///
/// That cancels the prologue, epilogue and link slot, all of which happen once per block,
/// leaving how many bytes one guest instruction costs to emit.
fn marginal(name: &str, body: &[u32]) {
    let program = |times: usize| -> Vec<u32> {
        let mut words = Vec::with_capacity(times * body.len() + 1);
        for _ in 0..times {
            words.extend_from_slice(body);
        }
        words.push(TAIL);
        words
    };
    let (small_bytes, small_count) = match block_bytes(&program(2)) {
        Some(value) => value,
        None => return println!("{name:28} unavailable"),
    };
    let (big_bytes, big_count) = match block_bytes(&program(5)) {
        Some(value) => value,
        None => return println!("{name:28} unavailable"),
    };
    let delta_bytes = big_bytes.saturating_sub(small_bytes) as f64;
    let delta_count = big_count.saturating_sub(small_count) as f64;
    if delta_count == 0.0 {
        return println!("{name:28} the difference is too small");
    }
    println!(
        "{name:28} marginal {:6.1} bytes/insn  ~ {:4.1} host insns",
        delta_bytes / delta_count,
        delta_bytes / delta_count / 4.2
    );
}

fn main() {
    absolute("b . (one instruction, pure fixed overhead)", &[TAIL]);
    println!();

    // Per-instruction cost (fill the block, discounting the fixed overhead directly)
    fill("add r0, r0, r1", 0xE080_0001);
    fill("add r0, r1, #7", 0xE281_0007);
    fill("adds r0, r0, r1", 0xE090_0001);
    fill("mov r0, r1", 0xE1A0_0001);
    fill("mov r0, r1, lsr #3", 0xE1A0_01A1);
    fill("and r0, r1, r2", 0xE001_0002);
    fill("ldr r0, [r1, #4]", 0xE591_0004);
    fill("str r0, [r1, #4]", 0xE581_0004);
    println!();

    // Four instructions per group in every family, repeated 2 and 5 times
    marginal("add r0, r0, r1", &[0xE080_0001; 4]);
    marginal(
        "add r0, r1, r2 (three distinct registers)",
        &[0xE081_0002; 4],
    );
    marginal("add r0, r1, #7", &[0xE281_0007; 4]);
    marginal("adds r0, r0, r1 (writes the flags)", &[0xE090_0001; 4]);
    marginal("muls r0, r1, r2", &[0xE010_0291; 4]);
    marginal("mov r0, r1, lsr #3 (shifted)", &[0xE1A0_01A1; 4]);
    marginal("ldr r0, [r1,#4]", &[0xE591_0004; 4]);
    marginal("str r0, [r1,#4]", &[0xE581_0004; 4]);
    marginal("push {r0,r1} (stmdb)", &[0xE92D_0003; 4]);
    marginal("push {r0-r3} (stmdb)", &[0xE92D_000F; 4]);
    marginal("push {r0-r7} (stmdb)", &[0xE92D_00FF; 4]);
    marginal("pop {r0-r7} (ldmia)", &[0xE8BD_00FF; 4]);

    // Conditional execution: subcs and orrcs after one cmp
    marginal(
        "cmp r0,r1 ; subcs r2,r1,r0",
        &[0xE150_0001, 0x2041_2000, 0xE1A0_0000, 0xE1A0_0000],
    );
    marginal(
        "cmp ; subcs (shifted operand)",
        &[0xE150_0001, 0x2040_00A1, 0xE1A0_0000, 0xE1A0_0000],
    );
}
