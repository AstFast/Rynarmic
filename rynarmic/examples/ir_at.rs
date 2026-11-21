//! Translate one block out of a plain ARM/Thumb image and print the IR.
//!
//! `dump_block_ir` reads a fixed bench blob; this reads whatever image the caller names, so a block
//! from a real pack can be looked at without a harness around it.
//!
//! Usage: `cargo run --example ir_at -- <image> <address> [arm|thumb] [--ops]`

use rynarmic::block::{self, Translation};
use rynarmic::memory::SimpleMemory;

const BASE: u32 = 0x1000_0000;
const SPAN: usize = 0x20_0000;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: ir_at <image> <address> [arm|thumb]");
    let addr = {
        let text = args.next().expect("an address is required");
        u32::from_str_radix(text.trim_start_matches("0x"), 16).expect("address must be hex")
    };
    let thumb = match args.next().as_deref() {
        Some("thumb") => true,
        Some("arm") | None => false,
        Some(other) => panic!("state must be arm or thumb, not {other}"),
    };
    let show_ops = std::env::args().any(|a| a == "--ops");

    let image = std::fs::read(&path).expect("the image has to be readable");
    assert!(image.len() <= SPAN, "image is larger than the mapping");
    let mut memory = SimpleMemory::new(BASE, SPAN);
    let offset = (addr - BASE) as usize;
    assert!(offset < image.len(), "the address is past the image");
    memory.write_bytes(BASE + offset as u32, &image[offset..]);

    let translated = match block::translate(&mut memory, addr, thumb) {
        Ok(Translation::Block(block)) => block,
        Ok(Translation::SingleStep) => return println!("{addr:#x}: single-step only"),
        Err(err) => return println!("{addr:#x}: decoding failed: {err}"),
    };
    println!(
        "{addr:#x} ({}): {} instruction(s) -> {} IR op(s), entry flags {:?}, next_pc={:#x}",
        if thumb { "thumb" } else { "arm" },
        translated.instruction_count,
        translated.ops.len(),
        translated.entry_flags,
        translated.next_pc
    );
    if show_ops {
        for (index, op) in translated.ops.iter().enumerate() {
            println!("  {index:>3}  {op:?}");
        }
    }
}
