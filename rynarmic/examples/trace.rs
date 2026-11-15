//! A small tracing tool: print the PC, the instruction and the key registers one at a time, for comparison against a reference implementation or real hardware.
//!
//! Usage: `cargo run --example trace`

use rynarmic::{Cpu, Memory, memory::SimpleMemory};

fn main() {
    let mut memory = SimpleMemory::new(0, 0x4000);
    // ARM state: a loop summing 10+9+...+1
    memory.write_words(
        0x1000,
        &[
            0xE3A0_0000, // mov r0, #0
            0xE3A0_100A, // mov r1, #10
            0xE080_0001, // add r0, r0, r1
            0xE251_1001, // subs r1, r1, #1
            0xE351_0000, // cmp r1, #0
            0x1AFF_FFFB, // bne 0x1008
            0xE12F_FF1E, // bx lr
        ],
    );
    memory.write_words(0x2000, &[0xEAFF_FFFE]); // an infinite loop

    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(0x1000);
    cpu.state.set_reg(14, 0x2000);

    for step in 0..64 {
        let pc = cpu.state.pc();
        let inst = cpu.memory.read_u32(pc);
        println!(
            "step {step:02} pc={pc:#06x} inst={inst:#010x} r0={:#04x} r1={:#04x} Z={}",
            cpu.state.reg(0),
            cpu.state.reg(1),
            cpu.state.cpsr.z()
        );
        if cpu.state.pc() == 0x2000 {
            break;
        }
        cpu.step().expect("unexpected exception");
    }
}
