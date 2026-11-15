//! A probe: print the machine code a stretch of instructions compiles to, to check what the emitter produced.
use rynarmic::{Memory, backend, block, memory::SimpleMemory};

const CODE: u32 = 0x1000;

fn dump(name: &str, words: &[u32]) {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(CODE, words);
    let translated = match block::translate(&mut memory, CODE, false) {
        Ok(block::Translation::Block(block)) => block,
        _ => return println!("{name}: cannot form a block"),
    };
    let window = memory.fast_window();
    let mut windows = [None; rynarmic::memory::MAX_WINDOWS];
    windows[0] = window;
    let mut arena = backend::CodeArena::new(1 << 20);
    let jit = backend::compile::<SimpleMemory>(&translated, false, windows, &mut arena).unwrap();
    let ptr = jit.entry_ptr();
    let len = jit.code_len();
    println!("{name}: {len} bytes");
    for chunk in 0..len.div_ceil(16) {
        print!("  {:04X}  ", chunk * 16);
        for i in 0..16 {
            let off = chunk * 16 + i;
            if off < len {
                print!("{:02X} ", unsafe { *ptr.add(off) });
            } else {
                print!("   ");
            }
        }
        println!();
    }
}

fn main() {
    // Two adds as a block of their own: prologue, two instructions, epilogue, so each add's shape is visible at a glance
    dump(
        "add r0, r0, r1 ×2",
        &[
            0xE080_0001, // add r0, r0, r1
            0xE080_0001,
            0xEAFF_FFFE, // b .
        ],
    );
    dump(
        "add r0, r1, #7 ×2",
        &[
            0xE281_0007, // add r0, r1, #7
            0xE281_0007,
            0xEAFF_FFFE,
        ],
    );
    dump("mov r0, r1 ×2", &[0xE1A0_0001, 0xE1A0_0001, 0xEAFF_FFFE]);
    let mut filled = vec![0xE1A0_0001u32; 31];
    filled.push(0xEAFF_FFFE);
    dump("mov r0, r1 x31 (fills the block)", &filled);
    // The udiv guard shape: expect to see the capture (0F 9x setcc) and cmovne (0F 45)
    dump(
        "cmp; subcs (shifted); orrcs; cmp (the udiv shape)",
        &[
            0xE150_0001, // cmp   r0, r1
            0x2041_20A0, // subcs r2, r1, r0, lsr #1
            0x2383_3001, // orrcs r3, r3, #1
            0xE150_0001, // cmp   r0, r1 (overwrites the flags at the chain's end)
            0xEAFF_FFFE, // b .
        ],
    );
}
