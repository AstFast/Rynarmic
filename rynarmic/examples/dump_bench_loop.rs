//! Triage aid: compile the benchmark's hot loop (a subs/bne back edge) and write the machine code to a file.
//!
//! Usage: `cargo run --release --example dump_bench_loop`
//! Then `objdump -D -b binary -m i386:x86-64 -M att bench_loop.bin`

use rynarmic::{Memory, backend, block, memory::SimpleMemory};

const CODE: u32 = 0x1000;
fn main() {
    let mut memory = SimpleMemory::new(0, 0x8000);
    memory.write_words(
        CODE,
        &[
            0xE250_0001, // subs r0, r0, #1
            0x1AFF_FFFD, // bne 0x1000
            0xEAFF_FFFE, // b .
        ],
    );
    let translated = match block::translate(&mut memory, CODE, false) {
        Ok(block::Translation::Block(block)) => block,
        _ => return println!("cannot form a block"),
    };
    let mut windows = [None; rynarmic::memory::MAX_WINDOWS];
    windows[0] = memory.fast_window();
    let mut arena = backend::CodeArena::new(1 << 20);
    let jit = backend::compile::<SimpleMemory>(&translated, false, windows, &mut arena).unwrap();
    let ptr = jit.entry_ptr();
    let len = jit.code_len();
    println!("benchmark hot-loop block: {len} bytes");
    let bytes: Vec<u8> = (0..len).map(|i| unsafe { *ptr.add(i) }).collect();
    for (j, chunk) in bytes.chunks(16).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02X}")).collect();
        println!("  {:04X}  {}", j * 16, hex.join(" "));
    }
    std::fs::write("target/bench_loop.bin", &bytes).unwrap();
    println!("wrote target/bench_loop.bin (disassemble it with objdump)");
}
