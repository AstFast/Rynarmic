//! Correctness of multiple flat windows (`Memory::fast_windows`).
//!
//! With one window, "write the hitting segment's base into `WINDOW`" is harmless (every segment is the same pointer),
//! so the defect only shows up with **two or more segments** when one block passes through the second segment and then takes a **compile-time constant address**,
//! because that constant address's offset is computed against segment 0.
//!
//! This test has that shape: `ldr [r1]` hits segment 1, and immediately afterwards `ldr [pc, #x]` folds into
//! a constant address inside segment 0. Before the fix the JIT read 0 (from segment 1 at that offset) while the interpreter read the right value.

use rynarmic::memory::{FastWindow, MAX_WINDOWS};
use rynarmic::{Cpu, Memory};

const CODE: u32 = 0x4000;
const CODE_LEN: usize = 0x1000;
const DATA_BASE: u32 = 0x8000;
const DATA_LEN: usize = 0x1000;
/// Where the constant pool lands (inside segment 0).
const POOL: u32 = 0x4100;

/// Two non-adjacent flat regions covering guest `[0x4000, 0x5000)` and `[0x8000, 0x9000)`.
///
/// Each region is allocated separately, so the host pointers necessarily differ — exactly the soil the defect grows in.
#[derive(Clone)]
struct TwoWindows {
    code: Box<[u8]>,
    data: Box<[u8]>,
}

impl TwoWindows {
    fn new() -> Self {
        Self {
            code: vec![0u8; CODE_LEN].into_boxed_slice(),
            data: vec![0u8; DATA_LEN].into_boxed_slice(),
        }
    }

    fn slice(&self, addr: u32) -> Option<(&[u8], usize)> {
        if (CODE..CODE + CODE_LEN as u32).contains(&addr) {
            return Some((&self.code, (addr - CODE) as usize));
        }
        if (DATA_BASE..DATA_BASE + DATA_LEN as u32).contains(&addr) {
            return Some((&self.data, (addr - DATA_BASE) as usize));
        }
        None
    }

    fn slice_mut(&mut self, addr: u32) -> Option<(&mut [u8], usize)> {
        if (CODE..CODE + CODE_LEN as u32).contains(&addr) {
            return Some((&mut self.code, (addr - CODE) as usize));
        }
        if (DATA_BASE..DATA_BASE + DATA_LEN as u32).contains(&addr) {
            return Some((&mut self.data, (addr - DATA_BASE) as usize));
        }
        None
    }

    fn write_word(&mut self, addr: u32, value: u32) {
        let (bytes, off) = self.slice_mut(addr).expect("write out of range");
        bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn peek_word(&self, addr: u32) -> u32 {
        let (bytes, off) = self.slice(addr).expect("read out of range");
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }
}

impl Memory for TwoWindows {
    fn read_u8(&mut self, addr: u32) -> u8 {
        self.slice(addr).map_or(0, |(bytes, off)| bytes[off])
    }

    fn write_u8(&mut self, addr: u32, value: u8) {
        if let Some((bytes, off)) = self.slice_mut(addr) {
            bytes[off] = value;
        }
    }

    fn fast_windows(&self) -> [Option<FastWindow>; MAX_WINDOWS] {
        let mut windows = [None; MAX_WINDOWS];
        windows[0] = Some(FastWindow {
            ptr: self.code.as_ptr() as *mut u8,
            base: CODE,
            len: CODE_LEN,
        });
        windows[1] = Some(FastWindow {
            ptr: self.data.as_ptr() as *mut u8,
            base: DATA_BASE,
            len: DATA_LEN,
        });
        windows
    }
}

/// `r2 = [r1]` (hitting segment 1) followed by `r0 = [pc, #0xf0]` (folding into a constant address in segment 0).
fn program() -> TwoWindows {
    let mut memory = TwoWindows::new();
    memory.write_word(CODE, 0xE3A0_1902); // mov r1, #0x8000
    memory.write_word(CODE + 4, 0xE591_2000); // ldr r2, [r1] -> segment 1
    memory.write_word(CODE + 8, 0xE59F_00F0); // ldr r0, [pc, #0xf0] → 0x4100
    memory.write_word(CODE + 12, 0xEAFF_FFFE); // b . -- so the block ends here
    memory.write_word(POOL, 0x1122_3344);
    // Segment 1's first word: deliberately different from the constant pool's
    memory.write_word(DATA_BASE, 0x5566_7788);
    memory
}

#[test]
fn second_window_then_absolute_load_reads_the_right_segment() {
    let mut cpu = Cpu::new(program());
    cpu.state.set_pc(CODE);
    // The whole block runs in one go: both loads happen in one JIT call, with segment 1 hitting first
    cpu.run_cached(4).expect("the block should run cleanly");

    assert_eq!(cpu.state.reg(2), 0x5566_7788, "the load from segment 1");
    assert_eq!(
        cpu.state.reg(0),
        0x1122_3344,
        "the constant-address load was polluted by the previous segment's window base"
    );
    assert!(
        cpu.compiled_block_count() > 0,
        "this path has to really take the JIT"
    );
}

/// A differential control of the same shape: the JIT and the interpreter must produce one result.
#[test]
fn jit_matches_interpreter_across_windows() {
    let mut jit = Cpu::new(program());
    jit.state.set_pc(CODE);
    jit.run_cached(4).expect("the jit path");

    let mut interp = Cpu::new(program());
    interp.state.set_pc(CODE);
    interp.set_jit_enabled(false);
    interp.run(4).expect("the single-step path");

    assert_eq!(jit.state.reg(0), interp.state.reg(0), "r0");
    assert_eq!(jit.state.reg(2), interp.state.reg(2), "r2");
}

/// The write direction must not be polluted either: `str` into segment 1, then `str` to a constant address in segment 0.
#[test]
fn store_to_second_window_does_not_poison_absolute_store() {
    let mut memory = TwoWindows::new();
    memory.write_word(CODE, 0xE3A0_1902); // mov r1, #0x8000
    memory.write_word(CODE + 4, 0xE3A0_2011); // mov r2, #0x11
    memory.write_word(CODE + 8, 0xE581_2000); // str r2, [r1] -> segment 1
    memory.write_word(CODE + 12, 0xE3A0_0022); // mov r0, #0x22
    memory.write_word(CODE + 16, 0xE58F_00E8); // str r0, [pc, #0xe8] → 0x4100
    memory.write_word(CODE + 20, 0xEAFF_FFFE); // b .

    let mut cpu = Cpu::new(memory.clone());
    cpu.state.set_pc(CODE);
    cpu.run_cached(6).expect("the block should run cleanly");

    assert_eq!(
        cpu.memory.peek_word(DATA_BASE),
        0x11,
        "the write into segment 1"
    );
    assert_eq!(
        cpu.memory.peek_word(POOL),
        0x22,
        "the write to the constant address"
    );
    assert!(
        cpu.compiled_block_count() > 0,
        "this path has to really take the JIT"
    );
}

/// The whole-range window check fast path for `LDM`/`STM`: when a window other than 0 hits, the base register
/// has to be that segment's host pointer, not the `WINDOW` (segment 0) the prologue loaded.
///
/// The shape comes from a real branch point in a shipped application: `ldmia r1!, {r4}` then `cmp r4, #0` then `beq`.
/// The data sits in segment 1 at `0x8100`, while segment 0's `0x4100` is deliberately written as 0 —
/// so with the wrong base `r4` becomes 0, `Z` sets and the branch is taken.
const SECOND_WINDOW_DATA: u32 = DATA_BASE + 0x100;
const FIRST_WINDOW_SHADOW: u32 = CODE + 0x100;

fn ldm_program() -> TwoWindows {
    let mut memory = TwoWindows::new();
    memory.write_word(CODE, 0xE3A0_1902); // mov   r1, #0x8000
    memory.write_word(CODE + 4, 0xE281_1C01); // add   r1, r1, #0x100
    memory.write_word(CODE + 8, 0xE8B1_0010); // ldmia r1!, {r4}
    memory.write_word(CODE + 12, 0xE354_0000); // cmp   r4, #0
    memory.write_word(CODE + 16, 0x0A00_0002); // beq   CODE + 0x20
    memory.write_word(CODE + 20, 0xE3A0_5001); // mov   r5, #1  (not taken)
    memory.write_word(CODE + 24, 0xEAFF_FFFE); // b     .
    memory.write_word(CODE + 28, 0xEAFF_FFFE); // b     .
    memory.write_word(CODE + 32, 0xE3A0_5003); // mov   r5, #3  (taken)
    memory.write_word(CODE + 36, 0xEAFF_FFFE); // b     .
    memory.write_word(SECOND_WINDOW_DATA, 0x0006_F440);
    memory.write_word(FIRST_WINDOW_SHADOW, 0);
    memory
}

#[test]
fn ldm_in_second_window_matches_interpreter() {
    let mut jit = Cpu::new(ldm_program());
    jit.state.set_pc(CODE);
    jit.run_cached(8).expect("the jit path");

    let mut interp = Cpu::new(ldm_program());
    interp.state.set_pc(CODE);
    interp.set_jit_enabled(false);
    interp.run(8).expect("the single-step path");

    assert!(
        jit.compiled_block_count() > 0,
        "this path has to really take the JIT"
    );
    assert_eq!(
        jit.state.reg(1),
        interp.state.reg(1),
        "r1's writeback matches the interpreter"
    );
    assert_eq!(
        jit.state.reg(4),
        interp.state.reg(4),
        "r4 has to be read from segment 1"
    );
    assert_eq!(
        jit.state.reg(5),
        interp.state.reg(5),
        "the CMP/BEQ branch outcome"
    );
    assert_eq!(jit.state.reg(1), 0x8104, "r1's writeback");
    assert_eq!(
        jit.state.reg(4),
        0x0006_F440,
        "r4 has to be read from segment 1"
    );
    assert_eq!(
        jit.state.reg(5),
        1,
        "a non-zero value must not let BEQ be taken"
    );
    assert_eq!(
        jit.state.cpsr.bits() & (1 << 30),
        interp.state.cpsr.bits() & (1 << 30),
        "the Z flag"
    );
}

#[test]
fn stm_in_second_window_writes_the_right_segment() {
    let mut memory = TwoWindows::new();
    memory.write_word(CODE, 0xE3A0_1902); // mov   r1, #0x8000
    memory.write_word(CODE + 4, 0xE281_1C01); // add   r1, r1, #0x100
    memory.write_word(CODE + 8, 0xE3A0_4077); // mov   r4, #0x77
    memory.write_word(CODE + 12, 0xE8A1_0010); // stmia r1!, {r4}
    memory.write_word(CODE + 16, 0xEAFF_FFFE); // b     .

    let mut jit = Cpu::new(memory.clone());
    jit.state.set_pc(CODE);
    jit.run_cached(5).expect("the jit path");

    let mut interp = Cpu::new(memory);
    interp.state.set_pc(CODE);
    interp.set_jit_enabled(false);
    interp.run(5).expect("the single-step path");

    assert!(
        jit.compiled_block_count() > 0,
        "this path has to really take the JIT"
    );
    assert_eq!(
        jit.memory.peek_word(SECOND_WINDOW_DATA),
        0x77,
        "the write has to land in segment 1"
    );
    assert_eq!(
        jit.memory.peek_word(SECOND_WINDOW_DATA),
        interp.memory.peek_word(SECOND_WINDOW_DATA),
        "matches the interpreter"
    );
    assert_eq!(
        jit.memory.peek_word(FIRST_WINDOW_SHADOW),
        0,
        "segment 0 should not be written"
    );
}
