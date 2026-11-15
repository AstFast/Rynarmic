//! Prints the block at an address in a real kernel **as it is**: the IR sequence plus whether the flags are still live after each position.
//!
//! The point is to answer "why is this hot block expensive": first the IR and guard counts, then flag liveness —
//! if `flags_dead_at` is true on the hot path, the emitter could have skipped the flags.
//!
//! Usage: `cargo run --release --example dump_block_ir -- 0x1168`

include!(concat!(env!("OUT_DIR"), "/compiled_bench.rs"));

use rynarmic::block::{self, Translation};
use rynarmic::memory::SimpleMemory;

fn main() {
    if !AVAILABLE {
        println!("no ARM cross toolchain, skipping.");
        return;
    }
    let pc = std::env::args()
        .nth(1)
        .map(|arg| {
            let arg = arg.trim_start_matches("0x");
            u32::from_str_radix(arg, 16).expect("the address must be written in hex")
        })
        .unwrap_or(0x1168);

    let mut memory = SimpleMemory::new(0, 0x10_0000);
    memory.write_bytes(BASE, BLOB);

    let translated = match block::translate(&mut memory, pc, false) {
        Ok(Translation::Block(block)) => block,
        Ok(Translation::SingleStep) => return println!("{pc:#x}: single-step only"),
        Err(err) => return println!("{pc:#x}: decoding failed: {err}"),
    };

    println!(
        "{pc:#x}: {} instructions -> {} IR ops, entry {:?}, next_pc={:#x}",
        translated.instruction_count,
        translated.ops.len(),
        translated.entry_flags,
        translated.next_pc
    );
    for (index, op) in translated.ops.iter().enumerate() {
        let dead = if translated.flags_dead_at(index + 1) {
            "   <- the flags are dead from here on"
        } else {
            ""
        };
        let guard = if matches!(op, rynarmic::IrOp::Cond { .. }) {
            "  [guard]"
        } else {
            ""
        };
        println!("{index:3}  {op:?}{guard}{dead}");
    }
}
