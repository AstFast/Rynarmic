//! Guest CPU state: the general registers, CPSR/SPSR, and the processor mode.

/// A processor mode (encoded in the low five bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    User = 0b1_0000,
    Fiq = 0b1_0001,
    Irq = 0b1_0010,
    Supervisor = 0b1_0011,
    Abort = 0b1_0111,
    Undefined = 0b1_1011,
    System = 0b1_1111,
}

impl Mode {
    pub const MASK: u32 = 0b1_1111;

    pub fn from_bits(value: u32) -> Mode {
        match value & Mode::MASK {
            0b1_0000 => Mode::User,
            0b1_0001 => Mode::Fiq,
            0b1_0010 => Mode::Irq,
            0b1_0011 => Mode::Supervisor,
            0b1_0111 => Mode::Abort,
            0b1_1011 => Mode::Undefined,
            _ => Mode::System,
        }
    }

    /// Whether this mode is privileged, i.e. may write the CPSR control bits and reach the banked
    /// registers.
    pub fn is_privileged(self) -> bool {
        !matches!(self, Mode::User)
    }
}

/// The program status register, held as a bitfield with the semantics of ARM's CPSR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpsr(u32);

impl Cpsr {
    pub const N: u32 = 1 << 31;
    pub const Z: u32 = 1 << 30;
    pub const C: u32 = 1 << 29;
    pub const V: u32 = 1 << 28;
    pub const Q: u32 = 1 << 27;
    pub const T: u32 = 1 << 5;
    pub const F: u32 = 1 << 6;
    pub const I: u32 = 1 << 7;
    pub const A: u32 = 1 << 8;

    #[inline]
    pub const fn from_bits(bits: u32) -> Cpsr {
        Cpsr(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    const fn get(self, mask: u32) -> bool {
        self.0 & mask != 0
    }

    #[inline]
    fn set(&mut self, mask: u32, value: bool) {
        if value {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }

    pub fn n(self) -> bool {
        self.get(Cpsr::N)
    }
    pub fn z(self) -> bool {
        self.get(Cpsr::Z)
    }
    pub fn c(self) -> bool {
        self.get(Cpsr::C)
    }
    pub fn v(self) -> bool {
        self.get(Cpsr::V)
    }
    pub fn t(self) -> bool {
        self.get(Cpsr::T)
    }
    pub fn i(self) -> bool {
        self.get(Cpsr::I)
    }
    pub fn f(self) -> bool {
        self.get(Cpsr::F)
    }
    /// The saturating-arithmetic overflow flag (ARMv5TE).
    pub fn q(self) -> bool {
        self.get(Cpsr::Q)
    }

    pub fn set_n(&mut self, value: bool) {
        self.set(Cpsr::N, value)
    }
    pub fn set_z(&mut self, value: bool) {
        self.set(Cpsr::Z, value)
    }
    pub fn set_c(&mut self, value: bool) {
        self.set(Cpsr::C, value)
    }
    pub fn set_v(&mut self, value: bool) {
        self.set(Cpsr::V, value)
    }
    pub fn set_t(&mut self, value: bool) {
        self.set(Cpsr::T, value)
    }
    pub fn set_i(&mut self, value: bool) {
        self.set(Cpsr::I, value)
    }
    pub fn set_f(&mut self, value: bool) {
        self.set(Cpsr::F, value)
    }
    pub fn set_q(&mut self, value: bool) {
        self.set(Cpsr::Q, value)
    }

    pub fn mode(self) -> Mode {
        Mode::from_bits(self.0)
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.0 = (self.0 & !Mode::MASK) | (mode as u32 & Mode::MASK);
    }

    /// Sets N and Z from a result — the part every arithmetic and logical operation shares.
    pub fn set_nz(&mut self, result: u32) {
        self.set_n(result & 0x8000_0000 != 0);
        self.set_z(result == 0);
    }

    /// Bitmask over the four condition flags N/Z/C/V.
    pub const NZCV: u32 = Cpsr::N | Cpsr::Z | Cpsr::C | Cpsr::V;
    /// The three flags a logical operation writes (V is left alone).
    pub const NZC: u32 = Cpsr::N | Cpsr::Z | Cpsr::C;

    /// Writes several flag bits at once.
    ///
    /// Compared with calling `set_*` per flag this is a single branchless merge. Flag updates sit
    /// on the hot path, where a boolean test per bit compiles into a run of conditional branches.
    #[inline]
    pub fn update_flags(&mut self, bits: u32, mask: u32) {
        self.0 = (self.0 & !mask) | (bits & mask);
    }

    /// Packs N/Z/C/V into their bit positions from a result plus carry and overflow — branchless.
    #[inline]
    pub fn pack_nzcv(result: u32, carry: bool, overflow: bool) -> u32 {
        let n = (result >> 31) & 1;
        let z = u32::from(result == 0);
        let c = u32::from(carry);
        let v = u32::from(overflow);
        (n << 31) | (z << 30) | (c << 29) | (v << 28)
    }
}

/// Deferred condition flags.
///
/// Only the arithmetic instructions that write all of NZCV (ADD/SUB/RSB/CMP/CMN) are recorded here
/// by the JIT. The N/Z/C/V in `CPSR` may be stale meanwhile, so a reader has to materialize first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LazyFlags {
    /// The result of the operation, kept so N/Z can be recovered from it.
    pub result: u32,
    /// bit31 = dirty, bit1 = V, bit0 = C, all in ARM terms.
    pub flags: u32,
}

impl LazyFlags {
    pub const DIRTY: u32 = 1 << 31;
    pub const V: u32 = 1 << 1;
    pub const C: u32 = 1;

    #[inline]
    pub fn is_dirty(self) -> bool {
        self.flags & Self::DIRTY != 0
    }

    #[inline]
    pub fn carry(self) -> bool {
        self.flags & Self::C != 0
    }

    #[inline]
    pub fn overflow(self) -> bool {
        self.flags & Self::V != 0
    }

    #[inline]
    pub fn set(&mut self, result: u32, carry: bool, overflow: bool) {
        self.result = result;
        self.flags = Self::DIRTY | u32::from(carry) | (u32::from(overflow) << 1);
    }

    #[inline]
    pub fn materialize(&self, cpsr: &mut Cpsr) {
        if self.is_dirty() {
            cpsr.update_flags(
                Cpsr::pack_nzcv(self.result, self.carry(), self.overflow()),
                Cpsr::NZCV,
            );
        }
    }
}

/// The architecturally visible state of an A32 core.
///
/// Note that there is one set of general registers and one SPSR: there are no mode banks at all.
/// The banked registers (the per-mode R13/R14 copies and the SPSR banks) arrive together with the
/// exception model; until then `MRS/MSR SPSR` read and write that single SPSR, and a host must not
/// expect a mode switch to swap stacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmState {
    pub r: [u32; 16],
    pub cpsr: Cpsr,
    pub spsr: Cpsr,
    /// The JIT's deferred flags; while dirty, the NZCV in `cpsr` is stale.
    pub lazy_flags: LazyFlags,
    /// The remaining instruction budget of a link chain — JIT bookkeeping that the interpreter
    /// never reads.
    ///
    /// A block subtracts its own instruction count on **entry** and checks for exhaustion just
    /// before jumping straight to its successor: that is how a chain cannot walk around
    /// `run_cached`'s `max_instructions`. At the end of a chain it may sit slightly below zero (the
    /// last block runs whole), and the driver derives the executed count from that.
    pub jit_budget: u32,
    /// The access-fault flag: 0 = fine, non-zero = this run hit an access fault.
    ///
    /// This is the slot the host is handed through
    /// [`crate::memory::Memory::set_fault_flag`]. Living in `ArmState` rather than in the host's
    /// memory object lets the JIT test it with one `cmp dword [state + FAULT], 0`: no register is
    /// spent on it and no indirect call is needed.
    pub fault: u32,
    /// The captured truth value of a condition (0 or 1).
    ///
    /// Used inside conditional-execution blocks: work under a guard can destroy the x86 flags (a
    /// shifted `subcs`, say) while the next guard still reads that same set. Rather than packing
    /// the flags into a descriptor (9–13 instructions), store the condition as one byte with
    /// `setcc` and let each later guard on the same condition test it (2 instructions). Which
    /// condition's truth value this holds is decided at compile time; at run time it is only 0/1.
    pub cond_cache: u32,
    /// A "pending exception" marker raised by JIT code: bit31 = an exception is pending, bit30 =
    /// breakpoint (otherwise SVC), low 24 bits = imm. Blocks containing SVC/BKPT now compile to
    /// machine code, and the driver reads this field after the block returns and clears it; the
    /// interpreter path does not go through it.
    ///
    /// Why a separate field rather than reusing `fault`: `fault` is the access-fault slot the host
    /// binds via [`crate::memory::Memory::set_fault_flag`], and its meaning belongs to the host.
    pub pending_exception: u32,
}

impl Default for ArmState {
    fn default() -> Self {
        let mut cpsr = Cpsr::default();
        cpsr.set_mode(Mode::Supervisor);
        Self {
            r: [0; 16],
            cpsr,
            spsr: Cpsr::default(),
            lazy_flags: LazyFlags::default(),
            jit_budget: 0,
            fault: 0,
            cond_cache: 0,
            pending_exception: 0,
        }
    }
}

impl ArmState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the deferred flags back into CPSR and clears the descriptor.
    #[inline]
    pub fn materialize_lazy_flags(&mut self) {
        if self.lazy_flags.is_dirty() {
            self.lazy_flags.materialize(&mut self.cpsr);
            self.lazy_flags = LazyFlags::default();
        }
    }

    #[inline]
    pub fn flags_are_lazy(&self) -> bool {
        self.lazy_flags.is_dirty()
    }

    #[inline]
    pub fn reg(&self, index: u8) -> u32 {
        self.r[index as usize]
    }

    #[inline]
    pub fn set_reg(&mut self, index: u8, value: u32) {
        self.r[index as usize] = value;
    }

    /// The address of the current instruction, i.e. PC before the pipeline offset is applied.
    #[inline]
    pub fn pc(&self) -> u32 {
        self.r[15]
    }

    #[inline]
    pub fn set_pc(&mut self, value: u32) {
        self.r[15] = value;
    }

    #[inline]
    pub fn thumb(&self) -> bool {
        self.cpsr.t()
    }
}
