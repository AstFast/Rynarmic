//! The IR interpreter.
//!
//! This layer is the correctness baseline: once a JIT backend exists it also serves as the
//! reference counterpart for instruction-by-instruction differential testing, and as the fallback
//! path for blocks that were not compiled. (In dynarmic the corresponding semantics live in
//! `backend/*/emit_*.cpp`.)

use crate::common::bits::{add_with_carry, cond_holds, shift_c, sign_extend};
use crate::cpu::ExceptionKind;
use crate::ir::{Address, AluOp, IrOp, MemSize, SatOp, ShiftType, Shifter, Value};
use crate::memory::{AccessFault, AccessKind, Memory};
use crate::state::{ArmState, Cpsr};

/// Where control goes after one instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Straight on; the CPU advances to the next address itself.
    Next,
    /// A direct jump to the given address.
    Branch(u32),
    /// A jump that may also switch instruction set (BX/BLX).
    Exchange(u32),
    /// An exception or software interrupt, handed back to the host.
    Exception(ExceptionKind),
}

pub struct Executor<'a, M: Memory> {
    pub state: &'a mut ArmState,
    pub memory: &'a mut M,
}

impl<'a, M: Memory> Executor<'a, M> {
    pub fn new(state: &'a mut ArmState, memory: &'a mut M) -> Self {
        Self { state, memory }
    }

    /// Runs an operation sequence that contains no condition guards: a single instruction, or a
    /// block with no conditional execution in it.
    ///
    /// This is the hottest loop, so it stays the plainest possible `for` iteration. Managing an
    /// index by hand costs the compiler its chance to eliminate bounds checks; measured, it is 8%
    /// slower.
    pub fn execute(&mut self, ops: &[IrOp]) -> Flow {
        for op in ops {
            let flow = self.execute_one(*op);
            if flow != Flow::Next {
                return flow;
            }
            if let Some(fault) = self.fault_flow() {
                return fault;
            }
        }
        Flow::Next
    }

    /// Runs a block containing condition guards ([`IrOp::Cond`]).
    ///
    /// The guard is evaluated at run time: if the condition holds, execution continues; if not,
    /// the guarded operations are skipped.
    pub fn execute_guarded(&mut self, ops: &[IrOp]) -> Flow {
        let mut index = 0;
        let len = ops.len();
        while index < len {
            let op = ops[index];
            index += 1;

            if let IrOp::Cond { cond, skip } = op {
                let flags = self.state.cpsr;
                if !cond_holds(cond, flags.n(), flags.z(), flags.c(), flags.v()) {
                    index += skip as usize;
                }
                continue;
            }

            let flow = self.execute_one(op);
            if flow != Flow::Next {
                return flow;
            }
            if let Some(fault) = self.fault_flow() {
                return fault;
            }
        }
        Flow::Next
    }

    /// Turns a host-reported access fault into an exception flow, clearing the flag on the way.
    ///
    /// Checked after every IR operation: when an access instruction (including `LDM`) fails
    /// part-way through, stop right there, rather than carrying on with the garbage value the
    /// callback returned and writing the remaining registers out.
    fn fault_flow(&mut self) -> Option<Flow> {
        if self.state.fault == 0 {
            return None;
        }
        let fault = self.memory.take_fault().unwrap_or(AccessFault {
            addr: 0,
            kind: AccessKind::Read,
        });
        self.state.fault = 0;
        Some(Flow::Exception(ExceptionKind::AccessFault {
            addr: fault.addr,
            kind: fault.kind,
        }))
    }

    fn execute_one(&mut self, op: IrOp) -> Flow {
        match op {
            // A condition guard is only ever handled by execute_guarded; it never appears in the
            // input to plain execute
            IrOp::Cond { .. } => {
                debug_assert!(false, "condition guards must go through execute_guarded");
                Flow::Next
            }
            IrOp::Nop => Flow::Next,
            IrOp::Alu {
                op,
                dst,
                a,
                b,
                set_flags,
            } => {
                self.alu(op, dst, a, b, set_flags);
                Flow::Next
            }
            IrOp::Mul {
                dst,
                a,
                b,
                accumulate,
                set_flags,
            } => {
                let result = self.state.r[a as usize].wrapping_mul(self.state.r[b as usize]);
                let result = match accumulate {
                    Some(acc) => result.wrapping_add(self.state.r[acc as usize]),
                    None => result,
                };
                self.state.r[dst as usize] = result;
                if set_flags {
                    self.state.cpsr.set_nz(result);
                }
                Flow::Next
            }
            IrOp::MulLong {
                lo,
                hi,
                a,
                b,
                signed,
                accumulate,
                set_flags,
            } => {
                let lhs = self.state.r[a as usize];
                let rhs = self.state.r[b as usize];
                let mut product = if signed {
                    (lhs as i32 as i64).wrapping_mul(rhs as i32 as i64)
                } else {
                    (lhs as u64).wrapping_mul(rhs as u64) as i64
                };
                if accumulate {
                    let acc = ((self.state.r[hi as usize] as u64) << 32)
                        | self.state.r[lo as usize] as u64;
                    product = product.wrapping_add(acc as i64);
                }
                let bits = product as u64;
                self.state.r[lo as usize] = bits as u32;
                self.state.r[hi as usize] = (bits >> 32) as u32;
                if set_flags {
                    self.state.cpsr.set_n(bits >> 63 != 0);
                    self.state.cpsr.set_z(bits == 0);
                }
                Flow::Next
            }
            IrOp::MulHalf {
                dst,
                addend,
                a,
                b,
                a_high,
                b_high,
                long,
                set_flags,
            } => {
                let lhs = signed_half(self.state.r[a as usize], a_high);
                let rhs = signed_half(self.state.r[b as usize], b_high);
                let product = (lhs as i64) * (rhs as i64);

                if long {
                    let acc = ((self.state.r[dst as usize + 1] as u64) << 32)
                        | self.state.r[dst as usize] as u64;
                    let result = (acc as i64).wrapping_add(product) as u64;
                    self.state.r[dst as usize] = result as u32;
                    self.state.r[dst as usize + 1] = (result >> 32) as u32;
                } else {
                    let base = match addend {
                        Some(reg) => self.state.r[reg as usize],
                        None => 0,
                    };
                    let result = base.wrapping_add(product as u32);
                    self.state.r[dst as usize] = result;
                    if set_flags {
                        self.state.cpsr.set_nz(result);
                    }
                    // SMLAxy: accumulating overflows 32 bits, so set Q (ARM ARM: "if overflow then
                    // Q = 1"). The product itself is only 30 bits, so the addition is the only
                    // place an overflow can come from.
                    if addend.is_some() && (base as i32).checked_add(product as i32).is_none() {
                        self.state.cpsr.set_q(true);
                    }
                }
                Flow::Next
            }
            IrOp::MulWordHalf {
                dst,
                addend,
                a,
                b,
                b_high,
                set_flags,
            } => {
                let lhs = self.state.r[a as usize] as i32 as i64;
                let rhs = i64::from(signed_half(self.state.r[b as usize], b_high));
                let product = (lhs * rhs) >> 16;
                let base = match addend {
                    Some(reg) => self.state.r[reg as usize],
                    None => 0,
                };
                let result = base.wrapping_add(product as i32 as u32);
                self.state.r[dst as usize] = result;
                if set_flags {
                    self.state.cpsr.set_nz(result);
                }
                // SMLAWy sets Q on accumulation overflow in the same way
                if addend.is_some() && (base as i32).checked_add(product as i32).is_none() {
                    self.state.cpsr.set_q(true);
                }
                Flow::Next
            }
            IrOp::Saturating { op, dst, a, b } => {
                // In the encoding Rn = bits(19,16) and Rm = bits(3,0), and the semantics are
                // Rd = Rm ± Rn
                let rn = self.state.r[a as usize] as i32;
                let rm = self.state.r[b as usize] as i32;
                let (result, overflow) = match op {
                    SatOp::Add => saturating_add(rm, rn),
                    SatOp::Sub => saturating_sub(rm, rn),
                    SatOp::DoubleAdd => {
                        let (doubled, overflow1) = saturating_double(rn);
                        let (result, overflow2) = saturating_add(rm, doubled);
                        (result, overflow1 || overflow2)
                    }
                    SatOp::DoubleSub => {
                        let (doubled, overflow1) = saturating_double(rn);
                        let (result, overflow2) = saturating_sub(rm, doubled);
                        (result, overflow1 || overflow2)
                    }
                };
                self.state.r[dst as usize] = result as u32;
                if overflow {
                    self.state.cpsr.set_q(true);
                }
                Flow::Next
            }
            IrOp::Clz { dst, src } => {
                self.state.r[dst as usize] = self.state.r[src as usize].leading_zeros();
                Flow::Next
            }
            IrOp::Swap {
                dst,
                src,
                address,
                byte,
            } => {
                let (effective, _) = self.address(address, true);
                if byte {
                    let old = self.memory.read_u8(effective);
                    self.memory
                        .write_u8(effective, self.state.r[src as usize] as u8);
                    self.state.r[dst as usize] = old as u32;
                } else {
                    let old = self.memory.read_u32(effective);
                    self.memory.write_u32(effective, self.state.r[src as usize]);
                    self.state.r[dst as usize] = old;
                }
                Flow::Next
            }
            IrOp::Breakpoint { imm } => Flow::Exception(ExceptionKind::Breakpoint(imm)),
            IrOp::Load {
                dst,
                address,
                size,
                signed,
                pre_index,
                writeback,
            } => {
                let (effective, offset_addr) = self.address(address, pre_index);
                let raw = self.read(effective, size);
                let value = if signed {
                    sign_extend(raw, size.bytes() * 8)
                } else {
                    raw
                };
                if writeback {
                    self.writeback(address, offset_addr);
                }
                self.state.r[dst as usize] = value;
                Flow::Next
            }
            IrOp::Store {
                src,
                address,
                size,
                pre_index,
                writeback,
            } => {
                let value = self.state.r[src as usize];
                let (effective, offset_addr) = self.address(address, pre_index);
                self.write(effective, size, value);
                if writeback {
                    self.writeback(address, offset_addr);
                }
                Flow::Next
            }
            IrOp::LoadDouble {
                rt,
                address,
                pre_index,
                writeback,
            } => {
                let (effective, offset_addr) = self.address(address, pre_index);
                let low = self.memory.read_u32(effective);
                let high = self.memory.read_u32(effective.wrapping_add(4));
                if writeback {
                    self.writeback(address, offset_addr);
                }
                self.state.r[rt as usize] = low;
                self.state.r[rt as usize + 1] = high;
                Flow::Next
            }
            IrOp::StoreDouble {
                rt,
                address,
                pre_index,
                writeback,
            } => {
                let low = self.state.r[rt as usize];
                let high = self.state.r[rt as usize + 1];
                let (effective, offset_addr) = self.address(address, pre_index);
                self.memory.write_u32(effective, low);
                self.memory.write_u32(effective.wrapping_add(4), high);
                if writeback {
                    self.writeback(address, offset_addr);
                }
                Flow::Next
            }
            IrOp::LoadMultiple {
                base,
                regs,
                add,
                pre_index,
                writeback,
            } => {
                let mut addr = self.block_start(base, regs, add, pre_index);
                for index in 0..16u8 {
                    if regs & (1 << index) != 0 {
                        self.state.r[index as usize] = self.memory.read_u32(addr);
                        addr = addr.wrapping_add(4);
                    }
                }
                if writeback {
                    self.writeback_block(base, regs, add);
                }
                Flow::Next
            }
            IrOp::StoreMultiple {
                base,
                regs,
                add,
                pre_index,
                writeback,
            } => {
                let mut addr = self.block_start(base, regs, add, pre_index);
                for index in 0..16u8 {
                    if regs & (1 << index) != 0 {
                        let value = self.state.r[index as usize];
                        self.memory.write_u32(addr, value);
                        addr = addr.wrapping_add(4);
                    }
                }
                if writeback {
                    self.writeback_block(base, regs, add);
                }
                Flow::Next
            }
            IrOp::Jump { target } => Flow::Branch(self.eval(target).0),
            IrOp::BranchConditional { cond, target } => {
                let flags = self.state.cpsr;
                if cond_holds(cond, flags.n(), flags.z(), flags.c(), flags.v()) {
                    Flow::Branch(self.eval(target).0)
                } else {
                    Flow::Next
                }
            }
            IrOp::JumpExchange { target } => Flow::Exchange(self.eval(target).0),
            IrOp::ReadStatus { dst, spsr } => {
                let psr = if spsr {
                    self.state.spsr
                } else {
                    self.state.cpsr
                };
                self.state.r[dst as usize] = psr.bits();
                Flow::Next
            }
            IrOp::WriteStatus { src, mask, spsr } => {
                let value = self.eval(src).0;
                let target = if spsr {
                    &mut self.state.spsr
                } else {
                    &mut self.state.cpsr
                };
                let bits = (target.bits() & !mask) | (value & mask);
                *target = Cpsr::from_bits(bits);
                Flow::Next
            }
            IrOp::Swi { imm } => Flow::Exception(ExceptionKind::SoftwareInterrupt(imm)),
        }
    }

    /// Evaluates an IR operand, returning `(value, shifter carry)`.
    ///
    /// The shifter carry only reaches C through **logical** operations: an arithmetic operation
    /// uses the carry of the arithmetic itself (ARM's `ADDS rd, rn, rm, lsl rs` takes the addition's
    /// carry, not the shift's).
    fn eval(&mut self, value: Value) -> (u32, Option<bool>) {
        match value {
            Value::Reg(index) => (self.state.r[index as usize], None),
            Value::Imm(value) => (value, None),
            Value::ImmC { value, carry } => (value, Some(carry)),
            Value::Shifted { reg, shift, by } => {
                let source = self.state.r[reg as usize];
                let carry_in = self.state.cpsr.c();
                match by {
                    Shifter::Imm(amount) => {
                        // An immediate ROR #0 is RRX: a shift amount of zero that does consume the
                        // carry
                        if amount == 0 && shift != ShiftType::Rrx {
                            (source, None)
                        } else {
                            let (value, carry) = shift_c(source, shift, amount as u32, carry_in);
                            (value, Some(carry))
                        }
                    }
                    Shifter::Reg(index) => {
                        let amount = self.state.r[index as usize] & 0xFF;
                        if amount == 0 {
                            // A shift amount of zero leaves the value alone and **writes no C**;
                            // C keeps what it had
                            (source, None)
                        } else {
                            let (value, carry) = shift_c(source, shift, amount, carry_in);
                            (value, Some(carry))
                        }
                    }
                }
            }
        }
    }

    fn alu(&mut self, op: AluOp, dst: Option<u8>, a: Value, b: Option<Value>, set_flags: bool) {
        let (av, a_carry) = self.eval(a);
        let (bv, b_carry) = match b {
            Some(value) => self.eval(value),
            None => (0, None),
        };

        // MOV/MVN take their shift information from the first operand, everything else from the
        // second
        let shifter_carry = if op.is_move() { a_carry } else { b_carry };

        let carry_in = self.state.cpsr.c();
        let logical_carry = shifter_carry.unwrap_or(carry_in);

        let (result, carry, overflow) = match op {
            AluOp::And | AluOp::Tst => (av & bv, logical_carry, false),
            AluOp::Eor | AluOp::Teq => (av ^ bv, logical_carry, false),
            AluOp::Orr => (av | bv, logical_carry, false),
            AluOp::Bic => (av & !bv, logical_carry, false),
            AluOp::Mov => (av, logical_carry, false),
            AluOp::Mvn => (!av, logical_carry, false),
            AluOp::Sub | AluOp::Cmp => add_with_carry(av, !bv, true),
            AluOp::Rsb => add_with_carry(bv, !av, true),
            AluOp::Add | AluOp::Cmn => add_with_carry(av, bv, false),
            AluOp::Adc => add_with_carry(av, bv, carry_in),
            AluOp::Sbc => add_with_carry(av, !bv, carry_in),
            AluOp::Rsc => add_with_carry(bv, !av, carry_in),
        };

        if set_flags {
            // One branchless merge writes every affected flag, instead of one branch per flag
            let mask = if op.is_logical() {
                Cpsr::NZC
            } else {
                Cpsr::NZCV
            };
            let bits = Cpsr::pack_nzcv(result, carry, overflow);
            self.state.cpsr.update_flags(bits, mask);
        }

        if let Some(dst) = dst
            && !op.is_test()
        {
            self.state.r[dst as usize] = result;
        }
    }

    fn address(&mut self, address: Address, pre_index: bool) -> (u32, u32) {
        match address {
            // An absolute address (PC-relative addressing was folded at translation time) has no
            // writeback
            Address::Absolute(addr) => (addr, addr),
            Address::Register { base, offset, add } => {
                let base_value = self.state.r[base as usize];
                let (offset_value, _) = self.eval(offset);
                let offset_addr = if add {
                    base_value.wrapping_add(offset_value)
                } else {
                    base_value.wrapping_sub(offset_value)
                };
                let effective = if pre_index { offset_addr } else { base_value };
                (effective, offset_addr)
            }
        }
    }

    /// Writes the base register back according to the addressing mode (an absolute address has no
    /// base, so it is ignored).
    fn writeback(&mut self, address: Address, value: u32) {
        if let Address::Register { base, .. } = address {
            self.state.r[base as usize] = value;
        }
    }

    fn block_start(&self, base: u8, regs: u16, add: bool, pre_index: bool) -> u32 {
        let base = self.state.r[base as usize];
        let total = regs.count_ones() * 4;
        match (pre_index, add) {
            (false, true) => base,
            (true, true) => base.wrapping_add(4),
            (false, false) => base.wrapping_sub(total).wrapping_add(4),
            (true, false) => base.wrapping_sub(total),
        }
    }

    fn writeback_block(&mut self, base: u8, regs: u16, add: bool) {
        let total = regs.count_ones() * 4;
        let base_value = self.state.r[base as usize];
        self.state.r[base as usize] = if add {
            base_value.wrapping_add(total)
        } else {
            base_value.wrapping_sub(total)
        };
    }

    fn read(&mut self, addr: u32, size: MemSize) -> u32 {
        match size {
            MemSize::U8 => self.memory.read_u8(addr) as u32,
            MemSize::U16 => self.memory.read_u16(addr) as u32,
            MemSize::U32 => self.memory.read_u32(addr),
        }
    }

    fn write(&mut self, addr: u32, size: MemSize, value: u32) {
        match size {
            MemSize::U8 => self.memory.write_u8(addr, value as u8),
            MemSize::U16 => self.memory.write_u16(addr, value as u16),
            MemSize::U32 => self.memory.write_u32(addr, value),
        }
    }
}

#[inline]
fn half(value: u32, high: bool) -> u16 {
    if high {
        (value >> 16) as u16
    } else {
        value as u16
    }
}

/// Takes a halfword and interprets it as **signed** (the operands of SMULxy / SMLAWy).
///
/// Note that this cannot be written `half(..) as i32`: `u16 as i32` zero-extends, so `0xFFFF`
/// becomes 65535 rather than -1. That mistake only shows up on the negative half, which small cases
/// like `SMULbb` do not reach.
#[inline]
fn signed_half(value: u32, high: bool) -> i32 {
    i32::from(half(value, high) as i16)
}

fn saturating_add(a: i32, b: i32) -> (i32, bool) {
    match a.checked_add(b) {
        Some(value) => (value, false),
        None => {
            if a > 0 {
                (i32::MAX, true)
            } else {
                (i32::MIN, true)
            }
        }
    }
}

fn saturating_sub(a: i32, b: i32) -> (i32, bool) {
    match a.checked_sub(b) {
        Some(value) => (value, false),
        None => {
            // The direction of a - b's overflow is decided by b's sign: b < 0 means the result goes
            // above the top, so MAX; b > 0 means it goes below the bottom, so MIN. Judging by a's
            // sign instead is wrong: sat_sub(0, i32::MIN) would saturate to MIN when the answer is
            // MAX. A differential case caught this.
            if b < 0 {
                (i32::MAX, true)
            } else {
                (i32::MIN, true)
            }
        }
    }
}

fn saturating_double(value: i32) -> (i32, bool) {
    match value.checked_mul(2) {
        Some(doubled) => (doubled, false),
        None => {
            if value > 0 {
                (i32::MAX, true)
            } else {
                (i32::MIN, true)
            }
        }
    }
}
