//! The access-fault channel and resuming after `SVC`: the two seams a host relies on when using Rynarmic as an HLE backend.
//!
//! 1. **Access faults**: when the host's memory implementation cannot serve an address inside `read_*`/`write_*`,
//!    it writes non-zero into the slot given to `Memory::set_fault_flag`, and the CPU must stop **at once** and hand back
//!    an `ExceptionKind::AccessFault`, rather than carrying on with whatever garbage the callback returned.
//! 2. **Resuming after SVC**: after an `SVC` the PC stops at the SVC itself with the state intact, so a host can set registers, push the PC
//!    past it, and run again to carry on — the semantics a runtime built on "every call is one SVC" depends on.
//!    (Continues the previous sentence.)

use rynarmic::{
    Cpu, ExceptionKind,
    memory::{AccessFault, AccessKind, Memory},
};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
/// An address deliberately left unmapped (page zero, so that a null pointer faults loudly)
const NOWHERE: u32 = 0x0000_0000;

/// Paged memory: `[CODE, DATA + 0x1000)` is mapped and anything else raises.
struct FaultMemory {
    base: u32,
    data: Vec<u8>,
    slot: Option<*mut u32>,
    last: Option<AccessFault>,
}

impl FaultMemory {
    fn new() -> Self {
        Self {
            base: CODE,
            data: vec![0; 0x2000],
            slot: None,
            last: None,
        }
    }

    fn offset(&self, addr: u32) -> Option<usize> {
        let offset = addr.wrapping_sub(self.base) as usize;
        (offset < self.data.len()).then_some(offset)
    }

    fn fail(&mut self, addr: u32, kind: AccessKind) {
        if let Some(slot) = self.slot {
            // SAFETY: the slot is bound by the `Cpu` before execution and points at its `ArmState::fault`,
            // which stays valid until this method returns (the `Cpu` does not move during a callback).
            unsafe { *slot = 1 };
        }
        if self.last.is_none() {
            self.last = Some(AccessFault { addr, kind });
        }
    }
}

impl Memory for FaultMemory {
    fn read_u8(&mut self, addr: u32) -> u8 {
        match self.offset(addr) {
            Some(offset) => self.data[offset],
            None => {
                self.fail(addr, AccessKind::Read);
                0
            }
        }
    }

    fn write_u8(&mut self, addr: u32, value: u8) {
        match self.offset(addr) {
            Some(offset) => self.data[offset] = value,
            None => self.fail(addr, AccessKind::Write),
        }
    }

    fn set_fault_flag(&mut self, slot: *mut u32) {
        self.slot = Some(slot);
    }

    fn take_fault(&mut self) -> Option<AccessFault> {
        self.last.take()
    }
}

fn cpu_with(words: &[u32]) -> Cpu<FaultMemory> {
    let mut memory = FaultMemory::new();
    for (i, word) in words.iter().enumerate() {
        let addr = CODE + (i as u32) * 4;
        for (k, byte) in word.to_le_bytes().iter().enumerate() {
            memory.write_u8(addr + k as u32, *byte);
        }
    }
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    cpu
}

/// Asserts that this is an access fault and returns `(addr, kind)`.
fn expect_fault(result: Result<impl std::fmt::Debug, ExceptionKind>) -> (u32, AccessKind) {
    match result {
        Err(ExceptionKind::AccessFault { addr, kind }) => (addr, kind),
        other => panic!("expected an AccessFault, got {other:?}"),
    }
}

/// A read of an unmapped address: the single-step path must stop exactly at that instruction.
#[test]
fn read_fault_interpreter_stops_at_the_instruction() {
    // ldr r0, [r1] -- r1 points at an unmapped page
    let mut cpu = cpu_with(&[0xE591_0000]);
    cpu.state.set_reg(1, NOWHERE);

    let (addr, kind) = expect_fault(cpu.step());
    assert_eq!(addr, NOWHERE);
    assert_eq!(kind, AccessKind::Read);
    assert_eq!(cpu.state.pc(), CODE, "PC stops at the failing instruction");
    assert_eq!(cpu.state.reg(0), 0, "a failed read must not write r0");
}

/// A write to an unmapped address: the JIT path must stop too, rather than dropping the write and carrying on.
#[test]
fn write_fault_stops_the_jit_block() {
    // mov r0, #1 ; str r0, [r1] ; add r2, r2, #1
    let mut cpu = cpu_with(&[0xE3A0_0001, 0xE581_0000, 0xE282_2001]);
    cpu.state.set_reg(1, NOWHERE);
    cpu.state.set_reg(2, 0);

    let (addr, kind) = expect_fault(cpu.run_cached(8));
    assert_eq!(addr, NOWHERE);
    assert_eq!(kind, AccessKind::Write);
    assert_eq!(
        cpu.state.reg(2),
        0,
        "no instruction after the fault may execute"
    );
    assert_eq!(
        cpu.state.pc(),
        CODE,
        "the JIT reports the block-start PC (a block keeps no per-instruction PC)"
    );
}

/// The fast path (already mapped) is unaffected: the fault slot stays 0 and behaviour matches having no channel at all.
#[test]
fn mapped_accesses_do_not_raise() {
    let mut cpu = cpu_with(&[0xE591_0000, 0xE282_2001, 0xEAFF_FFFE]);
    cpu.state.set_reg(1, DATA);
    cpu.state.set_reg(2, 0);
    cpu.memory.data[(DATA - CODE) as usize] = 0xAB;

    let executed = cpu.run_cached(8).unwrap();
    assert!(
        executed >= 2,
        "at least two instructions run (the `b .` spin afterwards eats the rest of the budget)"
    );
    assert_eq!(cpu.state.reg(0), 0xAB);
    assert_eq!(cpu.state.reg(2), 1);
    assert_eq!(cpu.state.fault, 0);
}

/// A fetch of an unmapped address: `AccessKind::Fetch`, with the PC pointing at the address that could not be read.
#[test]
fn fetch_fault_is_reported_as_fetch() {
    // b to an unmapped address
    let mut cpu = cpu_with(&[0xEA00_0000 | 0x1000]); // b +0x4000 → 0x5008
    let (addr, kind) = expect_fault(cpu.run_cached(4));
    assert_eq!(kind, AccessKind::Fetch);
    assert!(addr != 0, "the fetch address has to be reported");
}

/// Resuming after `SVC`: the PC stops at the SVC, a host may rewrite registers and memory, and running on carries forward.
#[test]
fn svc_yields_and_resumes() {
    // mov r0, #1 ; svc #0x123 ; add r0, r0, #41
    let mut cpu = cpu_with(&[0xE3A0_0001, 0xEF00_0123, 0xE280_0029, 0xEAFF_FFFE]);

    match cpu.run_cached(8) {
        Err(ExceptionKind::SoftwareInterrupt(imm)) => assert_eq!(imm, 0x123),
        other => panic!("expected SoftwareInterrupt(0x123), got {other:?}"),
    }
    assert_eq!(cpu.state.pc(), CODE + 4, "PC stops at the SVC itself");
    assert_eq!(
        cpu.state.reg(0),
        1,
        "the instruction before the SVC has taken effect"
    );

    // The host services the call: put the answer in r0 and push the PC past the SVC
    cpu.state.set_reg(0, 7);
    cpu.state.set_pc(CODE + 8);

    cpu.run_cached(4)
        .expect("execution should continue after resuming");
    assert_eq!(
        cpu.state.reg(0),
        7 + 41,
        "later instructions use the value the host returned"
    );
    assert!(cpu.state.pc() > CODE + 8, "the add has been passed");
}

/// The SVC's block has to really compile to machine code, with the exception carried back to the driver through `pending_exception`.
/// This is the witness that SVCs go through the JIT (ROADMAP §8 item 2): without a compiled block the test proves nothing.
#[test]
fn svc_block_is_compiled_and_reports_the_exception() {
    // mov r0, #1 ; svc #0x123 ; add r0, r0, #41 ; b .
    let mut cpu = cpu_with(&[0xE3A0_0001, 0xEF00_0123, 0xE280_0029, 0xEAFF_FFFE]);

    match cpu.run_cached(8) {
        Err(ExceptionKind::SoftwareInterrupt(imm)) => assert_eq!(imm, 0x123),
        other => panic!("expected SoftwareInterrupt(0x123), got {other:?}"),
    }
    assert!(
        cpu.compiled_block_count() > 0,
        "the SVC block has to be really compiled, or this test has no witness"
    );
    assert_eq!(cpu.state.pc(), CODE + 4, "PC stops at the SVC itself");
    assert_eq!(
        cpu.state.reg(0),
        1,
        "the instruction before the SVC has taken effect"
    );
    assert_eq!(
        cpu.state.pending_exception, 0,
        "the driver clears it on reading"
    );

    // The host services the call, pushes the PC past it, and execution continues
    cpu.state.set_pc(CODE + 8);
    cpu.run_cached(4)
        .expect("execution should continue after resuming");
    assert!(cpu.state.pc() > CODE + 8, "the SVC has been passed");
}

/// BKPT also goes through the JIT, with bit 30 of the marker telling a breakpoint from an SVC.
#[test]
fn bkpt_block_reports_the_breakpoint() {
    // bkpt #7 ; b .
    let mut cpu = cpu_with(&[0xE120_0077, 0xEAFF_FFFE]);
    match cpu.run_cached(4) {
        Err(ExceptionKind::Breakpoint(imm)) => assert_eq!(imm, 7),
        other => panic!("expected Breakpoint(7), got {other:?}"),
    }
    assert!(
        cpu.compiled_block_count() > 0,
        "the BKPT block has to be really compiled"
    );
    assert_eq!(cpu.state.pc(), CODE, "PC stops at the BKPT itself");
}

/// The same through the interpreter: `run` must stop at the SVC too, and resume.
#[test]
fn svc_resumes_on_the_interpreter_too() {
    let mut cpu = cpu_with(&[0xE3A0_0001, 0xEF00_0007, 0xE280_0029]);
    match cpu.run(8) {
        Err(ExceptionKind::SoftwareInterrupt(imm)) => assert_eq!(imm, 7),
        other => panic!("expected SoftwareInterrupt(7), got {other:?}"),
    }
    assert_eq!(cpu.state.pc(), CODE + 4);

    cpu.state.set_reg(0, 100);
    cpu.state.set_pc(CODE + 8);
    cpu.run(4).unwrap();
    assert_eq!(cpu.state.reg(0), 141);
}
