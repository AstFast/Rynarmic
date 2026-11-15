//! D1: external differential testing against unicorn (QEMU-derived), an external reference for correctness.
#![cfg(unicorn_ref)]
//!
//! All earlier correctness checks were **self-referential** (the JIT and the
//! interpreter share one semantics implementation), so a semantic bug -- such as
//! the SMULxy halfword sign-extension one -- makes both paths wrong and both green.
//! This test feeds the same random program to unicorn (QEMU's ARM core semantics)
//! and compares r0-r12/r14, CPSR NZCV+Q and a digest of the data area.
//!
//! unicorn is an optional differential reference for comparison experiments: the
//! build script links it and enables this file (`unicorn_ref` cfg) only when
//! `RYNARMIC_UNICORN` points at an unpacked package (see build.rs); otherwise the
//! whole file compiles to nothing.

use rynarmic::{Cpsr, Cpu, memory::Memory};
use std::collections::HashMap;
use std::os::raw::c_void;

/// Sparse demand-paged memory over the full 4 GB: reads of untouched pages return 0
/// and writes allocate a page that persists. Byte-for-byte identical to the semantics
/// of demand_map_hook on the unicorn side (map a zero page on a fault).
struct SparseMemory {
    pages: HashMap<u32, Box<[u8; 4096]>>,
}

impl SparseMemory {
    fn new() -> Self {
        Self {
            pages: HashMap::new(),
        }
    }

    fn write_words(&mut self, addr: u32, words: &[u32]) {
        for (i, word) in words.iter().enumerate() {
            self.write_u32(addr + (i as u32) * 4, *word);
        }
    }

    fn peek_u32(&self, addr: u32) -> u32 {
        self.pages
            .get(&(addr & !0xFFF))
            .map(|page| {
                let offset = (addr & 0xFFF) as usize;
                u32::from_le_bytes([
                    page[offset],
                    page[offset + 1],
                    page[offset + 2],
                    page[offset + 3],
                ])
            })
            .unwrap_or(0)
    }
}

impl rynarmic::memory::Memory for SparseMemory {
    fn read_u8(&mut self, addr: u32) -> u8 {
        self.pages
            .get(&(addr & !0xFFF))
            .map(|page| page[(addr & 0xFFF) as usize])
            .unwrap_or(0)
    }

    fn write_u8(&mut self, addr: u32, value: u8) {
        self.pages
            .entry(addr & !0xFFF)
            .or_insert_with(|| Box::new([0; 4096]))[(addr & 0xFFF) as usize] = value;
    }
}

const CODE: u32 = 0x1000; // Address 0 is the ARM low vector page; unicorn applies FETCH_PROT to it
const DATA_START: u32 = 0x2000;
const MEMORY_SIZE: usize = 0x8000;
const PROGRAM_LEN: usize = 120;
/// 200 cases by default (24k instructions, about 3 s); `UC_CASES` overrides it (the D1 acceptance line uses 100k).
fn cases() -> usize {
    std::env::var("UC_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200)
}
const STEPS: u64 = 120;

// ---------------------------------------------------------------------------
// unicorn FFI (unicorn2's C API, stable and tiny)
// ---------------------------------------------------------------------------

mod ffi {
    use std::os::raw::{c_int, c_void};

    pub const UC_ARCH_ARM: u32 = 1;
    pub const UC_MODE_ARM: u32 = 0;
    // arm.h: INVALID=0, APSR=1, APSR_NZCV=2, CPSR=3, R0=66 (Rn = 66+n, then SP/LR/PC).
    // Observed: LR=10 (right next to PC=11). 139 is an invalid id -- reads and writes
    // are no-ops (older versions always returned 0, masked by the BL-free generator;
    // that was the source of the deprecation warnings on every earlier run).
    pub const UC_ARM_REG_CPSR: c_int = 3;
    pub const UC_ARM_REG_LR: c_int = 10;
    pub const UC_ARM_REG_PC: c_int = 11;
    pub fn reg(n: u32) -> c_int {
        66 + n as c_int
    }

    unsafe extern "C" {
        pub fn uc_open(arch: u32, mode: u32, engine: &mut *mut c_void) -> c_int;
        pub fn uc_close(engine: *mut c_void) -> c_int;
        pub fn uc_mem_map(engine: *mut c_void, address: u64, size: usize, perms: u32) -> c_int;
        pub fn uc_mem_write(
            engine: *mut c_void,
            address: u64,
            bytes: *const u8,
            size: usize,
        ) -> c_int;
        pub fn uc_mem_read(engine: *mut c_void, address: u64, bytes: *mut u8, size: usize)
        -> c_int;
        pub fn uc_reg_write(engine: *mut c_void, regid: c_int, value: *const c_void) -> c_int;
        pub fn uc_reg_read(engine: *mut c_void, regid: c_int, value: *mut c_void) -> c_int;
        pub fn uc_emu_start(
            engine: *mut c_void,
            begin: u64,
            until: u64,
            timeout: u64,
            count: usize,
        ) -> c_int;
        pub fn uc_hook_add(
            engine: *mut c_void,
            hook: *mut usize,
            hook_type: c_int,
            callback: UcMemHook,
            user_data: *mut c_void,
            begin: u64,
            end: u64,
        ) -> c_int;
        // Declare the same C symbol once with each callback signature: rustc's
        // clashing_extern_declarations lint warns about "different signatures",
        // but that is exactly the intent here.
        #[allow(clippy::similar_names, clashing_extern_declarations)]
        #[link_name = "uc_hook_add"]
        pub fn uc_hook_add_code(
            engine: *mut c_void,
            hook: *mut usize,
            hook_type: c_int,
            callback: UcCodeHook,
            user_data: *mut c_void,
            begin: u64,
            end: u64,
        ) -> c_int;
    }

    /// Callback signature for UC_HOOK_CODE (used only for triage traces).
    pub type UcCodeHook = unsafe extern "C" fn(
        engine: *mut std::os::raw::c_void,
        address: u64,
        size: u32,
        user_data: *mut c_void,
    );

    thread_local! {
        /// Recent PC trace (for triage).
        pub static TRACE: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    pub unsafe extern "C" fn code_trace_hook(
        _engine: *mut std::os::raw::c_void,
        address: u64,
        _size: u32,
        _user_data: *mut c_void,
    ) {
        TRACE.with(|t| {
            let mut t = t.borrow_mut();
            if t.len() < 512 {
                t.push(address as u32);
            }
        });
    }

    pub const UC_HOOK_CODE: c_int = 1 << 2;

    /// Memory hook callback signature (unicorn2 mem hook: shared by read/write/fetch unmapped).
    pub type UcMemHook = unsafe extern "C" fn(
        engine: *mut std::os::raw::c_void,
        mem_type: c_int,
        address: u64,
        size: c_int,
        value: i64,
        user_data: *mut c_void,
    ) -> bool;

    /// UC_HOOK_MEM_UNMAPPED = mask of READ|WRITE|FETCH_UNMAPPED.
    pub const UC_HOOK_MEM_UNMAPPED: c_int = (1 << 4) | (1 << 5) | (1 << 6);

    /// Demand-mapping hook: in unicorn2 returning true from a mem hook does not
    /// actually skip the access (it ends up as UC_ERR_MAP); the supported approach
    /// is to `uc_mem_map` the missing page from inside the hook.
    /// Mapped pages start zeroed and writes persist -- strictly matching the
    /// semantics of `SparseMemory` below.
    pub unsafe extern "C" fn demand_map_hook(
        engine: *mut std::os::raw::c_void,
        _mem_type: c_int,
        address: u64,
        _size: c_int,
        _value: i64,
        _user_data: *mut c_void,
    ) -> bool {
        let page = address & !0xFFF;
        // Map only the current page: the hook fires only for unmapped pages, so a
        // single-page mapping never overlaps an existing one; a cross-page access
        // makes unicorn fire the hook again for the second page.
        let rc = unsafe { uc_mem_map(engine, page, 0x1000, 7) };
        eprintln!("demand_map_hook: addr={address:#x} type={_mem_type} rc={rc}");
        rc == 0
    }
}

/// An open unicorn ARM instance.
struct Unicorn {
    engine: *mut std::os::raw::c_void,
    hook: usize,
}

impl Unicorn {
    fn new(program: &[u32], init_regs: &[u32; 13], cpsr: u32) -> Result<Self, i32> {
        let mut engine = std::ptr::null_mut();
        let err = unsafe { ffi::uc_open(ffi::UC_ARCH_ARM, ffi::UC_MODE_ARM, &mut engine) };
        if err != 0 {
            return Err(err);
        }
        let uc = Self { engine, hook: 0 };
        unsafe {
            assert_eq!(ffi::uc_mem_map(uc.engine, 0, MEMORY_SIZE, 7), 0); // 7 = UC_PROT_ALL
            // The top page of the 4 GB space must be mapped up front: a mid-run
            // uc_mem_map of the top page returns rc=0 but has no effect
            // (observed with a probe: ldmda reading 0xFFFFFFF4 gave UC_ERR_MAP; err=0 after pre-mapping).
            assert_eq!(ffi::uc_mem_map(uc.engine, 0xFFFF_F000, 0x1000, 7), 0);
            let mut code_hook = 0usize;
            assert_eq!(
                ffi::uc_hook_add_code(
                    uc.engine,
                    &mut code_hook,
                    ffi::UC_HOOK_CODE,
                    ffi::code_trace_hook,
                    std::ptr::null_mut(),
                    1,
                    0,
                ),
                0
            );
            assert_eq!(
                ffi::uc_hook_add(
                    uc.engine,
                    &uc.hook as *const usize as *mut usize,
                    ffi::UC_HOOK_MEM_UNMAPPED,
                    ffi::demand_map_hook,
                    std::ptr::null_mut(),
                    1, // begin != end -> the whole address space
                    0,
                ),
                0
            );
            let bytes = program_to_bytes(program);
            assert_eq!(
                ffi::uc_mem_write(uc.engine, CODE as u64, bytes.as_ptr(), bytes.len()),
                0
            );
            for (n, value) in init_regs.iter().enumerate() {
                assert_eq!(
                    ffi::uc_reg_write(
                        uc.engine,
                        ffi::reg(n as u32),
                        value as *const u32 as *const _
                    ),
                    0
                );
            }
            assert_eq!(
                ffi::uc_reg_write(
                    uc.engine,
                    ffi::UC_ARM_REG_CPSR,
                    &cpsr as *const u32 as *const _
                ),
                0
            );
            // PC must be written explicitly: the full run sets it as a side effect of
            // uc_emu_start(begin), while the single-step comparison starts from uc.pc();
            // the default 0 would execute nops from the vector page.
            let pc0 = CODE;
            assert_eq!(
                ffi::uc_reg_write(
                    uc.engine,
                    ffi::UC_ARM_REG_PC,
                    &pc0 as *const u32 as *const _
                ),
                0
            );
        }
        Ok(uc)
    }

    /// Run `count` instruction slots (a count=1 loop drives them one by one).
    ///
    /// We cannot use uc_emu_start(count=N) for the whole run: unicorn's count only
    /// includes actually executed instructions, not conditionally skipped ones,
    /// whereas our run() budget counts skipped slots too -- case 336 therefore
    /// showed a false "one step behind at the end" divergence. Driving one slot at
    /// a time keeps both sides aligned.
    fn run(&mut self, count: u64) -> i32 {
        for _ in 0..count {
            let begin = self.pc() as u64;
            let err = unsafe { ffi::uc_emu_start(self.engine, begin, begin + 4, 0, 1) };
            if err != 0 {
                return err;
            }
        }
        0
    }

    fn pc(&self) -> u32 {
        let mut value = 0u32;
        unsafe {
            ffi::uc_reg_read(
                self.engine,
                ffi::UC_ARM_REG_PC,
                &mut value as *mut u32 as *mut _,
            )
        };
        value
    }

    fn reg(&self, n: u32) -> u32 {
        let mut value = 0u32;
        unsafe { ffi::uc_reg_read(self.engine, ffi::reg(n), &mut value as *mut u32 as *mut _) };
        value
    }

    fn cpsr(&self) -> u32 {
        let mut value = 0u32;
        unsafe {
            ffi::uc_reg_read(
                self.engine,
                ffi::UC_ARM_REG_CPSR,
                &mut value as *mut u32 as *mut _,
            )
        };
        value
    }

    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn memory_digest(&self) -> u64 {
        let mut buf = vec![0u8; MEMORY_SIZE - DATA_START as usize];
        unsafe {
            ffi::uc_mem_read(self.engine, DATA_START as u64, buf.as_mut_ptr(), buf.len());
        }
        buf.chunks_exact(4).fold(0u64, |acc, c| {
            acc.rotate_left(5)
                .wrapping_add(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64)
        })
    }
}

impl Drop for Unicorn {
    fn drop(&mut self) {
        unsafe { ffi::uc_close(self.engine) };
    }
}

impl Unicorn {
    /// Read a 32-bit word from unicorn memory (for attribution: distinguish the original program from self-modified code).
    fn word_at(&self, addr: u32) -> u32 {
        let mut buf = [0u8; 4];
        unsafe {
            ffi::uc_mem_read(self.engine, addr as u64, buf.as_mut_ptr(), buf.len());
        }
        u32::from_le_bytes(buf)
    }
}

/// Reinterpret `&[u32]` as `&[u8]` (no external crate).
fn program_to_bytes(program: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(program.as_ptr() as *const u8, program.len() * 4) }
}

// ---------------------------------------------------------------------------
// Random program generator: same shape as tests/differential.rs (a separate copy to avoid cross-test coupling)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn reg(&mut self) -> u32 {
        (self.next() % 13) as u32
    }
}

fn build_program(rng: &mut Rng) -> Vec<u32> {
    (0..PROGRAM_LEN)
        .map(|index| {
            let rd = rng.reg();
            let rn = rng.reg();
            let rm = rng.reg();
            match rng.next() % 26 {
                0..=13 => {
                    // Data processing. ARM imposes SBZ (should-be-zero) fields and qemu
                    // rejects encodings that violate them:
                    // - MOV/MVN: rn must be 0; TST/TEQ/CMP/CMN: rd must be 0 and S must be 1
                    // Variants: condition codes (EQ/NE, 25%) and register-shift forms (25%)
                    let op = (rng.next() % 16) as u32;
                    let opcode = op << 21;
                    let is_move = op == 13 || op == 15;
                    let is_test = (8..=11).contains(&op);
                    let set_flags = if is_test || rng.next().is_multiple_of(2) {
                        1 << 20
                    } else {
                        0
                    };
                    let cond = match rng.next() % 4 {
                        0 => 0x0 << 28, // EQ
                        1 => 0x1 << 28, // NE
                        _ => 0xE << 28, // AL
                    };
                    let rn = if is_move { 0 } else { rn };
                    let rd = if is_test { 0 } else { rd };
                    if rng.next().is_multiple_of(2) {
                        let rot = (rng.next() % 16) as u32;
                        let imm8 = (rng.next() % 256) as u32;
                        // The DP immediate form must set bit25 (0x0200_0000): otherwise
                        // imm8 bit4=1 lands the encoding in the halfword/doubleword space
                        // and it is flagged UNPREDICTABLE
                        // (observed with an assembler: mov r0,#0xD0 = E3A000D0, base E200/E3A0).
                        cond | 0x0200_0000u32
                            | opcode
                            | set_flags
                            | rn << 16
                            | rd << 12
                            | rot << 8
                            | imm8
                    } else if rng.next().is_multiple_of(4) {
                        // Register-shift form: bit4=1, bit7=0, the shift amount comes from register Rs
                        let rs = rng.reg();
                        let shift = (rng.next() % 4) as u32;
                        // The base must not include 0xE000_0000: it already carries the
                        // cond=AL bits, and OR-ing with NE(0x1<<28) yields 0xF<<28
                        // (the NV space, which qemu rejects).
                        cond | opcode
                            | set_flags
                            | rn << 16
                            | rd << 12
                            | rm
                            | rs << 8
                            | shift << 5
                            | 1 << 4
                    } else {
                        let shift = (rng.next() % 4) as u32;
                        let amount = (rng.next() % 32) as u32;
                        cond | opcode
                            | set_flags
                            | rn << 16
                            | rd << 12
                            | rm
                            | shift << 5
                            | amount << 7
                    }
                }
                14 | 15 => {
                    // Multiply. MUL/MLA (60%) or long multiplies UMULL/SMULL/UMLAL/SMLAL (40%).
                    // Long multiplies require RdHi = RdLo+1 (otherwise UNPREDICTABLE, qemu rejects).
                    if rng.next() % 5 < 3 {
                        let acc = rng.next().is_multiple_of(2);
                        let ra = rng.reg();
                        let mut base = 0xE000_0000u32 | rd << 16 | rn << 8 | 0x90 | rm;
                        if acc {
                            base |= ra << 12 | 1 << 21;
                        }
                        base
                    } else {
                        let rdlo = (rng.next() % 12) as u32 & !1;
                        let signed = rng.next().is_multiple_of(2);
                        let acc = rng.next().is_multiple_of(2);
                        // Long-multiply opcode selection in field20 = 0x08|0x04(signed)|0x02(acc):
                        // bit23 stays 1, otherwise the encoding lands in undefined space and qemu rejects it.
                        let mut base = 0xE000_0000u32
                            | 0x08 << 20
                            | (rdlo + 1) << 16
                            | rdlo << 12
                            | rn << 8
                            | 0x90
                            | rm;
                        if signed {
                            base |= 1 << 22;
                        }
                        if acc {
                            base |= 1 << 21;
                        }
                        base
                    }
                }
                16 | 17 => {
                    // Single access: word/byte x immediate/register offset x direction x pre/post indexing.
                    // P=0,W=1 is UNPREDICTABLE and is never generated; an LDR with writeback needs rd != rn.
                    let load = rng.next().is_multiple_of(2);
                    let byte = !rng.next().is_multiple_of(4);
                    let reg_offset = rng.next().is_multiple_of(2);
                    let up = rng.next().is_multiple_of(2);
                    let pre = rng.next().is_multiple_of(2);
                    let wb = if pre && rng.next().is_multiple_of(2) {
                        1 << 21
                    } else {
                        0
                    };
                    let mut rd = rd;
                    if load && wb != 0 && rd == rn {
                        rd = (rd + 1) % 13;
                    }
                    let mut word = (if load { 1 << 20 } else { 0 })
                        | (if byte { 1 << 22 } else { 0 })
                        | (if up { 1 << 23 } else { 0 })
                        | (if pre { 1 << 24 } else { 0 })
                        | wb
                        | rn << 16
                        | rd << 12;
                    if reg_offset {
                        word |= 0xE600_0000u32 | rm;
                    } else {
                        let imm12 = (rng.next() % 0x100) as u32;
                        word |= 0xE400_0000u32 | imm12;
                    }
                    word
                }
                18 | 19 => {
                    // LDM/STM: four modes x writeback (50%), two registers in the list, excluding the base
                    // (otherwise UNPREDICTABLE, qemu rejects it).
                    let load = rng.next().is_multiple_of(2);
                    let mode = (rng.next() % 4) as u32; // DA/IA/DB/IB
                    let wb = if rng.next().is_multiple_of(2) {
                        1 << 21
                    } else {
                        0
                    };
                    let a = rng.reg();
                    let mut b = rng.reg();
                    if b == a {
                        b = (b + 1) % 13;
                    }
                    let mask = (1u32 << a) | (1u32 << b);
                    let mut base = rn;
                    if mask & (1 << base) != 0 {
                        // Base collides with the list: pick a non-conflicting base
                        base = (0..13u32).find(|r| mask & (1 << r) == 0).unwrap();
                    }
                    let mut word = (mode & 1) << 23 | (mode >> 1) << 24;
                    word = 0xE800_0000u32
                        | word
                        | (if load { 1 << 20 } else { 0 })
                        | wb
                        | base << 16
                        | mask;
                    word
                }
                20 | 21 => {
                    // Halfword-multiply family (operand fields calibrated against unicorn and an assembler):
                    // SMULxy 40% / SMLAxy 30% / SMULWy 15% / SMLALxy 15%
                    // (SMLALxy requires RdHi=RdLo+1 and an even RdLo)
                    let pick = rng.next() % 20;
                    let rdhi = (rng.next() % 12) as u32 & !1;
                    if pick < 8 {
                        // SMULxy: Rd(19-16), Rm(11-8), Rn(3-0), y=bit6, x=bit5
                        let xy = ((rng.next() % 4) as u32) << 5;
                        0xE160_0000u32 | rd << 16 | rm << 8 | 0x80 | xy | rn
                    } else if pick < 14 {
                        // SMLAxy: + Ra(15-12), accumulation overflow sets Q
                        let xy = ((rng.next() % 4) as u32) << 5;
                        0xE100_0000u32 | rd << 16 | rng.reg() << 12 | rm << 8 | 0x80 | xy | rn
                    } else if pick < 17 {
                        // SMULWy: full word Rn=bits(3,0), y halfword Rm=bits(11,8), Ra=0
                        let y = if rng.next().is_multiple_of(2) {
                            1 << 6
                        } else {
                            0
                        };
                        0xE120_0000u32 | rd << 16 | rm << 8 | 0xA0 | y | rn
                    } else if pick < 19 {
                        // SMLAWy: + Ra, accumulation overflow sets Q
                        let y = if rng.next().is_multiple_of(2) {
                            1 << 6
                        } else {
                            0
                        };
                        0xE120_0000u32 | rd << 16 | rng.reg() << 12 | rm << 8 | 0x80 | y | rn
                    } else {
                        // SMLALxy: RdLo(15-12)+RdHi(19-16)=RdLo+1
                        0xE140_0000u32 | (rdhi + 1) << 16 | rdhi << 12 | rm << 8 | 0x80 | rn
                    }
                }
                22 => {
                    // Saturating arithmetic: QADD/QSUB/QDADD/QDSUB
                    // (opcode selection in field20 = bits(23,20); bits 7-4 are always 0101;
                    // the third syntax operand Rm is in bits(19-16), Rd in 15-12, the second Rn in 3-0).
                    let op = match rng.next() % 4 {
                        0 => 0x10, // QADD
                        1 => 0x12, // QSUB
                        2 => 0x14, // QDADD
                        _ => 0x16, // QDSUB
                    };
                    0xE100_0000u32 | op << 20 | rm << 16 | rd << 12 | 0x50 | rn
                }
                _ => {
                    // Branches: b/Beq/Bne (60%) + BL (40%, writes LR, target inside the program)
                    let target = (rng.next() % PROGRAM_LEN as u64) as i32;
                    let imm24 = ((target - index as i32 - 2) & 0x00FF_FFFF) as u32;
                    let opcode = match rng.next() % 5 {
                        0 => 0x0A00_0000u32,     // beq
                        1 => 0x1A00_0000u32,     // bne
                        2 | 3 => 0xEB00_0000u32, // bl
                        _ => 0xEA00_0000u32,     // b
                    };
                    opcode | imm24
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main differential test
// ---------------------------------------------------------------------------

fn rynarmic_run(program: &[u32], seed: u64) -> Cpu<SparseMemory> {
    let mut rng = Rng(seed);
    let mut memory = SparseMemory::new();
    memory.write_words(CODE, program);
    let mut cpu = Cpu::new(memory);
    cpu.state.set_pc(CODE);
    for reg in 0..13u8 {
        let value = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
        cpu.state.set_reg(reg, value);
    }
    let flags = rng.next() as u32 & 0xF000_0000;
    cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() | flags);
    let _ = cpu.run(STEPS);
    cpu
}

/// Snapshot of one unicorn run: handle + r0-r12 + CPSR + LR; errors are (code, PC, actual word at PC).
type UcRun = Result<(Unicorn, [u32; 13], u32, u32), (i32, u32, u32)>;

fn unicorn_run(program: &[u32], seed: u64) -> UcRun {
    let mut rng = Rng(seed);
    let mut init_regs = [0u32; 13];
    for slot in &mut init_regs {
        *slot = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
    }
    let flags = rng.next() as u32 & 0xF000_0000;
    let cpsr = 0x13 | flags; // SVC mode + random NZCV
    let mut uc = Unicorn::new(program, &init_regs, cpsr).map_err(|e| (e, 0, 0))?;
    ffi::TRACE.with(|t| t.borrow_mut().clear());
    let err = uc.run(STEPS);
    if err != 0 {
        let pc = uc.pc();
        let trace_tail: Vec<String> = ffi::TRACE.with(|t| {
            t.borrow()
                .iter()
                .rev()
                .take(6)
                .rev()
                .map(|p| format!("{p:#x}"))
                .collect()
        });
        eprintln!("unicorn trace tail: {trace_tail:?}");
        let actual = if pc >= CODE && pc < CODE + PROGRAM_LEN as u32 * 4 {
            uc.word_at(pc)
        } else {
            0
        };
        return Err((err, pc, actual));
    }
    let mut regs = [0u32; 13];
    for (n, slot) in regs.iter_mut().enumerate() {
        *slot = uc.reg(n as u32);
    }
    let lr = {
        let mut value = 0u32;
        unsafe {
            ffi::uc_reg_read(
                uc.engine,
                ffi::UC_ARM_REG_LR,
                &mut value as *mut u32 as *mut _,
            )
        };
        value
    };
    let cpsr = uc.cpsr();
    Ok((uc, regs, cpsr, lr))
}

/// QADD family semantics probe: determine Rn/Rm direction and which operand is doubled (critical for QDSUB/QDADD).
#[test]
#[ignore = "triage tool"]
fn probe_saturating() {
    // r1 = 0x40000000 (bits 3-0 candidate), r2 = 0xC0000000 (bits 19-16 candidate), r3 = 0
    for (name, word) in [
        ("qsub  r0, r1, r2", 0xE122_0051u32),
        ("qdadd r0, r1, r2", 0xE142_0051u32),
        ("qdsub r0, r1, r2", 0xE162_0051u32),
        ("qadd  r0, r1, r2", 0xE102_0051u32),
    ] {
        let program = vec![word];
        let mut regs = [0x2000u32; 13];
        regs[1] = 0x4000_0000;
        regs[2] = 0xC000_0000;
        let mut uc = Unicorn::new(&program, &regs, 0x13).unwrap();
        let err = uc.run(4);
        let uc_r0 = uc.reg(0);
        let uc_q = uc.cpsr() & 0x0800_0000 != 0;

        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, 0x4000_0000);
        cpu.state.set_reg(2, 0xC000_0000);
        cpu.state.cpsr = Cpsr::from_bits(0x13);
        let _ = cpu.step();
        let my_r0 = cpu.state.r[0];
        let my_q = cpu.state.cpsr.bits() & 0x0800_0000 != 0;

        // Spec: QDSUB/QDADD/QSUB/QADD Rd, Rn, Rm -- Rn = bits(3,0), Rm = bits(19,16)
        let (rn, rm) = (0x4000_0000_u32 as i32, 0xC000_0000_u32 as i32);
        let spec = match name {
            n if n.starts_with("qadd") => rn.saturating_add(rm),
            n if n.starts_with("qsub") => rn.saturating_sub(rm),
            n if n.starts_with("qdadd") => rn.saturating_add(rm.saturating_mul(2)),
            _ => rn.saturating_sub(rm.saturating_mul(2)),
        } as u32;
        eprintln!(
            "{name}: word={word:08X} err={err}\n  spec={spec:08x} unicorn={uc_r0:08x}(Q:{uc_q}) ours={my_r0:08x}(Q:{my_q})"
        );
    }

    // Reproduce case 154 in place: QDSUB r6, r3, r10 with Rn=0 and Rm=0x84002388
    // (2 x Rm saturates to INT_MIN, then 0 - INT_MIN saturates to INT_MAX).
    {
        let word = 0xE16A_6053u32;
        let program = vec![word];
        let mut regs = [0x2000u32; 13];
        regs[10] = 0x8400_2388;
        let mut uc = Unicorn::new(&program, &regs, 0x13).unwrap();
        let err = uc.run(4);
        let uc_r0 = uc.reg(0);
        let uc_q = uc.cpsr() & 0x0800_0000 != 0;

        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(10, 0x8400_2388);
        cpu.state.cpsr = Cpsr::from_bits(0x13);
        let _ = cpu.step();
        eprintln!(
            "qdsub reproduction (Rn=0, Rm=0x84002388): err={err}\n  spec=7fffffff unicorn={uc_r0:08x}(Q:{uc_q}) ours={:08x}(Q:{})",
            cpu.state.r[6],
            cpu.state.cpsr.q()
        );
    }
}

/// Minimal probe: under count=1 driving, does a conditionally skipped slot consume count?
#[test]
#[ignore = "triage tool"]
fn probe_cond_skip() {
    // Z=1: movne is skipped. Drive 3 slots one at a time: if a skipped slot consumes
    // count then r1=3; otherwise unicorn stalls on movne and r1 is never set to 3.
    let program = vec![
        0xE3A0_0001u32, // mov r0, #1
        0x13A0_0002u32, // movne r0, #2 (Z=1 -> skipped)
        0xE3A0_1003u32, // mov r1, #3
    ];
    let regs = [0x2000u32; 13];
    let uc = Unicorn::new(&program, &regs, 0x13 | 0x4000_0000).unwrap(); // Z=1
    let mut slots = 0;
    for _ in 0..3 {
        let begin = uc.pc() as u64;
        let err = unsafe { ffi::uc_emu_start(uc.engine, begin, begin + 4, 0, 1) };
        if err != 0 {
            eprintln!("probe: err={err}");
            break;
        }
        slots += 1;
        let pc = uc.pc();
        eprintln!(
            "slot {slots}: pc={pc:#x} r0={:08x} r1={:08x}",
            uc.reg(0),
            uc.reg(1)
        );
    }
    eprintln!(
        "probe_cond_skip: after 3 driven slots r1={:08x} (3 = skipped slots consume count normally)",
        uc.reg(1)
    );
}

/// Minimal probe: locate the real source of UC_ERR_MAP (the instruction itself or hook/mapping interaction).
#[test]
#[ignore = "triage tool"]
fn probe_map_errors() {
    let cases: Vec<(&str, u32, u32)> = vec![
        (
            "stmda r0,{r4,r6} base in mapped region",
            0xE800_0050,
            0x3000,
        ),
        (
            "ldmia r0,{r4,r6} base in mapped region",
            0xE890_0050,
            0x3000,
        ),
        (
            "ldmda r0,{r4,r6} base in mapped region",
            0xE810_0050,
            0x3000,
        ),
        (
            "stmdb r0,{r4,r6} base in mapped region",
            0xE920_0050,
            0x3000,
        ),
        ("str r0,[r0],#-6 base in mapped region", 0xE400_0006, 0x3000),
        (
            "ldmda genuinely out of bounds (two in one page)",
            0xE810_0050,
            0xC3E0_0000,
        ),
        (
            "stmdb genuinely out of bounds (two in one page)",
            0xE920_0050,
            0xC3DF_FFFC,
        ),
        ("ldmda cross-page", 0xE810_0050, 0xC3E0_0004),
        ("str genuinely out of bounds", 0xE400_0006, 0xC3E0_0000),
        ("ldr genuinely out of bounds", 0xE410_0006, 0xC3E0_0000),
        ("ldmda wraps 4GB (r0=4)", 0xE810_0050, 0x4),
        ("ldmda wraps 4GB (r0=8)", 0xE810_0050, 0x8),
        ("ldr wraps 4GB (r0=0)", 0xE410_0006, 0x0),
        (
            "ldmda last word at top address (r0=0xFFFF_FFFC)",
            0xE810_0050,
            0xFFFF_FFFC,
        ),
    ];
    for (name, word, base) in cases {
        let program = vec![word];
        let mut regs = [0x2000u32; 13];
        regs[0] = base;
        let mut uc = match Unicorn::new(&program, &regs, 0x13) {
            Ok(uc) => uc,
            Err(e) => {
                eprintln!("{name}: uc_open failed with {e}");
                continue;
            }
        };
        let err = uc.run(8);
        let r3 = uc.reg(3);
        let r4 = uc.reg(4);
        eprintln!("{name}: word={word:08X} base={base:#x} -> err={err} r3={r3:08x} r4={r4:08x}");
    }

    // Top-page pre-mapping experiment: map [0xFFFFF000, 0x100000000) first, then access the top word
    let program = vec![0xE810_0050u32]; // ldmda r0, {r3, r7}
    let regs = [0xFFFF_FFFCu32; 13];
    let mut uc = Unicorn::new(&program, &regs, 0x13).unwrap();
    let rc = unsafe { ffi::uc_mem_map(uc.engine, 0xFFFF_F000, 0x1000, 7) };
    let err = uc.run(8);
    eprintln!(
        "after top-page pre-mapping, ldmda r0={{r3,r7}} (r0=0xFFFFFFFC): map rc={rc} err={err}"
    );
}

/// Per-instruction comparison: for mismatching cases, single-step both sides and report the first divergence.
#[test]
#[ignore = "triage tool"]
fn first_divergence() {
    let limit = std::env::var("UC_DIVERGENCE_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mut reported = 0usize;
    let cases = cases();
    for case in 0..cases {
        let seed = 0x5EED_0000 + case as u64;
        let mut rng = Rng(seed);
        let program = build_program(&mut rng);

        // Run the full comparison first, then dig only into mismatching cases
        let full = rynarmic_run(&program, seed);
        let mismatch = match unicorn_run(&program, seed) {
            Ok((_uc, uc_regs, uc_cpsr, uc_lr)) => {
                let mut m = Vec::new();
                for (n, (mine, ucv)) in full.state.r.iter().zip(uc_regs.iter()).enumerate() {
                    if mine != ucv {
                        m.push(format!("r{n}"));
                    }
                }
                if full.state.r[14] != uc_lr {
                    m.push("r14".into());
                }
                let mask = 0xF800_0000u32;
                if full.state.cpsr.bits() & mask != uc_cpsr & mask {
                    m.push("cpsr".into());
                }
                m
            }
            Err(_) => vec!["uc-error".into()],
        };
        if mismatch.is_empty() || reported >= limit {
            continue;
        }
        reported += 1;

        // ---- Single-step replay ----
        let mut rng = Rng(seed);
        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        let mut init_regs = [0u32; 13];
        for slot in &mut init_regs {
            *slot = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
        }
        for (r, value) in init_regs.iter().enumerate() {
            cpu.state.set_reg(r as u8, *value);
        }
        let flags = rng.next() as u32 & 0xF000_0000;
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() | flags);

        let uc = match Unicorn::new(&program, &init_regs, 0x13 | flags) {
            Ok(uc) => uc,
            Err(e) => {
                println!("case {case}: uc_open failed with {e}");
                continue;
            }
        };

        for step in 0..STEPS {
            let pc = cpu.state.pc();
            let word = program
                .get(((pc.wrapping_sub(CODE)) / 4) as usize)
                .copied()
                .unwrap_or(0);

            // Our single step
            let mine_before: [u32; 13] = cpu.state.r[..13].try_into().unwrap();
            let step_result = cpu.step();
            let mine_after: [u32; 13] = cpu.state.r[..13].try_into().unwrap();
            let mine_cpsr = cpu.state.cpsr.bits() & 0xF800_0000;
            let _ = mine_before;

            // unicorn single step
            let begin = uc.pc();
            let uerr =
                unsafe { ffi::uc_emu_start(uc.engine, begin as u64, (begin + 4) as u64, 0, 1) };
            if uerr != 0 {
                println!(
                    "case {case}: step {step} pc={pc:#x} word={word:08X} unicorn single step failed with {uerr} (ours had no exception: {step_result:?})"
                );
                break;
            }
            let mut uc_after = [0u32; 13];
            for (n, slot) in uc_after.iter_mut().enumerate() {
                *slot = uc.reg(n as u32);
            }
            let uc_cpsr = uc.cpsr() & 0xF800_0000;

            let mut diff = Vec::new();
            for n in 0..13 {
                if mine_after[n] != uc_after[n] {
                    diff.push(format!(
                        "r{n}: ours {:08x} (was {:08x}) vs uc {:08x}",
                        mine_after[n], mine_before[n], uc_after[n]
                    ));
                }
            }
            if mine_cpsr != uc_cpsr {
                diff.push(format!("cpsr: ours {mine_cpsr:08x} vs uc {uc_cpsr:08x}"));
            }
            if !diff.is_empty() {
                let before: Vec<String> = mine_before
                    .iter()
                    .enumerate()
                    .map(|(n, v)| format!("r{n}={v:08x}"))
                    .collect();
                println!(
                    "case {case}: first divergence at step {step} pc={pc:#x} word={word:08X}\n  {}\n  our state before divergence: {}",
                    diff.join("\n  "),
                    before.join(" ")
                );
                break;
            }
            if step_result.is_err() {
                println!(
                    "case {case}: ours raised {step_result:?} at step {step}, unicorn did not"
                );
                break;
            }
        }
    }
    println!("first-divergence report done, reported {reported} cases");
}

/// SMULxy semantics probe: compare unicorn behaviour / the ARM spec / our model for known operands.
#[test]
#[ignore = "triage tool"]
fn probe_smul() {
    // rn = 0x12345678, rm = 0x9ABCDEF0
    // SMULxy: Rd = half(Rn,x) * half(Rm,y); half is a signed halfword (B=[15:0], T=[31:16])
    let rn: i32 = 0x1234_5678u32 as i32;
    let rm: i32 = 0x9ABC_DEF0u32 as i32;
    let half = |v: i32, top: bool| -> i32 {
        if top {
            (v >> 16) as i16 as i32
        } else {
            (v as i16) as i32
        }
    };
    let variants: Vec<(&str, u32, bool, bool)> = vec![
        // (name, bits7-4, x=T?, y=T?)
        ("smulbb", 0x80, false, false),
        ("smulbt", 0x90, false, true),
        ("smultb", 0xA0, true, false),
        ("smultt", 0xE0, true, true),
    ];
    for (name, bits74, x, y) in variants {
        // SMULxy Rd, Rn, Rm: Rd=r0(19-16), Rn=r1(3-0), Rm=r2(11-8)
        let word = 0xE160_0000u32 | 2 << 8 | bits74 | 1;
        let program = vec![word];

        // unicorn
        let mut regs = [0x2000u32; 13];
        regs[1] = 0x1234_5678;
        regs[2] = 0x9ABC_DEF0;
        let mut uc = Unicorn::new(&program, &regs, 0x13 | 0x4000_0000).unwrap(); // preset C=1
        let err = uc.run(4);
        let uc_r0 = uc.reg(0);
        let uc_c = uc.cpsr() & 0x4000_0000 != 0;
        let uc_q = uc.cpsr() & 0x0800_0000 != 0;

        // Ours
        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, 0x1234_5678);
        cpu.state.set_reg(2, 0x9ABC_DEF0);
        cpu.state.cpsr = Cpsr::from_bits(0x13 | 0x4000_0000);
        let _ = cpu.step();
        let my_r0 = cpu.state.r[0];
        let my_c = cpu.state.cpsr.bits() & 0x4000_0000 != 0;
        let my_q = cpu.state.cpsr.bits() & 0x0800_0000 != 0;

        let spec = half(rn, x).wrapping_mul(half(rm, y)) as u32;
        eprintln!(
            "{name}: word={word:08X} err={err}\n  spec={spec:08x} unicorn={uc_r0:08x}(C kept:{uc_c} Q:{uc_q}) ours={my_r0:08x}(C kept:{my_c} Q:{my_q})"
        );
    }

    // SMULWy / SMLAWy: determine which field supplies the y halfword (bits 3-0 or 11-8)
    // r1 = 0x12345678 (full-word candidate), r2 = 0x9ABCDEF0
    for (name, word) in [
        ("smlawy(y=T)", 0xE120_0040u32 | 3 << 16 | 2 << 8 | 1), // bits5-4=00, y=bit6=1, Ra=r3=0x2000
        ("smulwy(y=T)", 0xE120_02E1u32),                        // bit5=1(M), y=bit6=1, ra=0
        (
            "smulwy(y=B)",
            0xE120_02A1u32, // bit5=1(M), y=bit6=0
        ),
        ("smlawy(y=B)", 0xE120_0001u32 | 3 << 16 | 2 << 8),
    ] {
        let program = vec![word];
        let mut regs = [0x2000u32; 13];
        regs[1] = 0x1234_5678;
        regs[2] = 0x9ABC_DEF0;
        let mut uc = Unicorn::new(&program, &regs, 0x13 | 0x4000_0000).unwrap();
        let err = uc.run(4);
        let uc_r0 = uc.reg(0);
        let uc_q = uc.cpsr() & 0x0800_0000 != 0;

        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, 0x1234_5678);
        cpu.state.set_reg(2, 0x9ABC_DEF0);
        cpu.state.cpsr = Cpsr::from_bits(0x13 | 0x4000_0000);
        let _ = cpu.step();
        let my_r0 = cpu.state.r[0];
        let my_q = cpu.state.cpsr.bits() & 0x0800_0000 != 0;

        // Candidate semantics: A = full word from bits(3-0)=r1 and y halfword from bits(11-8)=r2; B = the reverse
        let cand_a_t = ((rn as i64 * i64::from(half(rm, true))) >> 16) as u32;
        let cand_a_b = ((rn as i64 * i64::from(half(rm, false))) >> 16) as u32;
        let cand_b_t = ((rm as i64 * i64::from(half(rn, true))) >> 16) as u32;
        let cand_b_b = ((rm as i64 * i64::from(half(rn, false))) >> 16) as u32;
        eprintln!(
            "{name}: word={word:08X} err={err}\n  unicorn={uc_r0:08x}(Q:{uc_q}) ours={my_r0:08x}(Q:{my_q}) | A.top={cand_a_t:08x} A.bottom={cand_a_b:08x} B.top={cand_b_t:08x} B.bottom={cand_b_b:08x}"
        );
    }
}

/// JIT divergence bisector: find the smallest N at which "JIT runs N instructions
/// then the interpreter finishes" first differs from the pure-interpreter final state
/// (select a single case with UC_CASES).
#[test]
#[ignore = "triage tool"]
fn jit_bisect() {
    let case: u64 = std::env::var("UC_CASES")
        .expect("set UC_CASES=<case number>")
        .parse()
        .expect("UC_CASES must be a number");
    let seed = 0x5EED_0000 + case as u64;
    let mut rng = Rng(seed);
    let program = build_program(&mut rng);

    let build = |jit: bool| {
        let mut rng = Rng(seed);
        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        for reg in 0..13u8 {
            let value = DATA_START + ((rng.next() % 0x1FFF) as u32 & !3);
            cpu.state.set_reg(reg, value);
        }
        let flags = rng.next() as u32 & 0xF000_0000;
        cpu.state.cpsr = Cpsr::from_bits(cpu.state.cpsr.bits() | flags);
        cpu.set_jit_enabled(jit);
        cpu
    };

    let snapshot = |cpu: &Cpu<SparseMemory>| -> (Vec<u32>, u32, u32) {
        (cpu.state.r.to_vec(), cpu.state.cpsr.bits(), cpu.state.pc())
    };

    let pure_interp = {
        let mut cpu = build(false);
        let _ = cpu.run(STEPS);
        snapshot(&cpu)
    };

    // Bisect the smallest N: JIT runs N instructions + interpreter finishes != pure interpreter
    let differs = |n: u64| -> bool {
        let mut cpu = build(true);
        let _ = cpu.run(n);
        cpu.set_jit_enabled(false);
        let _ = cpu.run(STEPS - n);
        snapshot(&cpu) != pure_interp
    };
    let mut lo = 1u64;
    let mut hi = STEPS;
    if !differs(hi) {
        println!(
            "case {case}: JIT agrees with the interpreter throughout (the final-state difference may come from another mechanism)"
        );
        return;
    }
    while lo < hi {
        let mid = (lo + hi) / 2;
        if differs(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let n = lo;
    // Replay: inspect PC and registers at the end of the JIT segment
    let mut cpu = build(true);
    let _ = cpu.run(n - 1);
    let pc_before = cpu.state.pc();
    let _ = cpu.run(1); // one more instruction (completed inside the JIT)
    let (regs, cpsr, pc_after) = snapshot(&cpu);
    println!(
        "case {case}: minimum JIT divergence at N={n}, PC (end of JIT segment)={pc_before:#x} -> {pc_after:#x}, instruction={:08X?} (index {})\npure-interpreter final r9={:08x} cpsr={:08x}\nafter JIT r9={:08x} cpsr={:08x}",
        program
            .get(((pc_before.wrapping_sub(CODE)) / 4) as usize)
            .copied(),
        (pc_before.wrapping_sub(CODE) / 4) as usize,
        pure_interp.0[9],
        pure_interp.1,
        regs[9],
        cpsr,
    );
    let _ = &pure_interp.2;
}

// ---------------------------------------------------------------------------
// D2: coverage statistics (instruction families actually executed x hit counts)
// ---------------------------------------------------------------------------

/// Classify an instruction word into a readable family key (granularity: mnemonic + key variant bits).
fn classify(inst: u32) -> String {
    const DP_OPS: [&str; 16] = [
        "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "tst", "teq", "cmp", "cmn", "orr",
        "mov", "bic", "mvn",
    ];
    let field20 = (inst >> 20) & 0xFF;
    let bits2725 = (inst >> 25) & 7;
    let s_flag = (inst >> 20) & 1 == 1;
    let dp_op = ((field20 >> 1) & 0xF) as usize; // opcode = bits 24-21

    match bits2725 {
        // Multiply branches must come before DP: SMULxy's bit4=0 collides with the
        // DP register form, and QADD's bits7-4=0101 collides with the DP ASR variant.
        // Multiply: MUL/MLA/long multiply (bits 27-24 = 0000, bits 7-4 = 1001).
        0b000 if (inst & 0x0F00_0090) == 0x0000_0090 => {
            let acc = (inst >> 21) & 1 == 1;
            let long = (inst >> 23) & 1 == 1;
            match (long, acc) {
                (false, false) => format!("mul{}", if s_flag { "s" } else { "" }),
                (false, true) => format!("mla{}", if s_flag { "s" } else { "" }),
                (true, false) => {
                    if (inst >> 22) & 1 == 1 {
                        "smull".into()
                    } else {
                        "umull".into()
                    }
                }
                (true, true) => {
                    if (inst >> 22) & 1 == 1 {
                        "smlal".into()
                    } else {
                        "umlal".into()
                    }
                }
            }
        }
        // Halfword multiply: SMLAxy/SMLALxy/SMULxy/SMLAWy/SMULWy (bit7=1, bit4=0)
        0b000 if (inst & 0x90) == 0x80 => match field20 & 0x1F {
            0x10 => "smlaxy".into(),
            0x14 => "smlalxy".into(),
            0x16 => {
                let x = if (inst >> 5) & 1 == 1 { 't' } else { 'b' };
                let y = if (inst >> 6) & 1 == 1 { 't' } else { 'b' };
                format!("smul.{x}{y}")
            }
            0x12 => {
                if (inst >> 5) & 1 == 1 {
                    "smulwy".into()
                } else {
                    "smlawy".into()
                }
            }
            _ => format!("unknown.f{field20:02X}.p{:02X}", (inst >> 4) & 0xFF),
        },
        // Saturating arithmetic: QADD/QSUB/QDADD/QDSUB (bits 7-4 = 0101, field20 in 0x10-0x16)
        // (the overlap with DP register-shift ASR only occurs in the UNPREDICTABLE
        // region of "test instructions with S=0", which the generator never emits;
        // a field20 allowlist is therefore sufficient).
        0b000 if (inst & 0xF0) == 0x50 => match field20 & 0x1F {
            0x10 => "qadd".into(),
            0x12 => "qsub".into(),
            0x14 => "qdadd".into(),
            0x16 => "qdsub".into(),
            _ => format!("unknown.f{field20:02X}.p{:02X}", (inst >> 4) & 0xFF),
        },
        // Data processing: 001 = immediate, 000 with bit4=0 = immediate-shift register,
        // 000 with bit4=1 and bit7=0 = register-shift form.
        0b001 => format!("dp.{}.imm{}", DP_OPS[dp_op], if s_flag { "s" } else { "" }),
        0b000 if (inst >> 4) & 1 == 0 => {
            format!("dp.{}.reg{}", DP_OPS[dp_op], if s_flag { "s" } else { "" })
        }
        0b000 if (inst >> 7) & 1 == 0 => {
            format!(
                "dp.{}.regshift{}",
                DP_OPS[dp_op],
                if s_flag { "s" } else { "" }
            )
        }
        // Single access: 010 = immediate offset, 011 = register offset
        0b010 | 0b011 => {
            let mnemonic = if (inst >> 20) & 1 == 1 { "ldr" } else { "str" };
            let width = if (inst >> 22) & 1 == 1 { "b" } else { "" };
            let form = match ((inst >> 24) & 1, (inst >> 23) & 1, (inst >> 21) & 1) {
                (1, 1, 0) => "pre.up",
                (1, 0, 0) => "pre.down",
                (1, 1, 1) => "pre.up.wb",
                (1, 0, 1) => "pre.down.wb",
                (0, 1, _) => "post.up",
                (0, 0, _) => "post.down",
                _ => "odd",
            };
            let offset = if bits2725 == 0b010 { ".imm" } else { ".reg" };
            format!("{mnemonic}{width}{offset}.{form}")
        }
        // Multiple access: 100
        0b100 => {
            let mnemonic = if (inst >> 20) & 1 == 1 { "ldm" } else { "stm" };
            let mode = match ((inst >> 24) & 1, (inst >> 23) & 1) {
                (0, 0) => "da",
                (0, 1) => "ia",
                (1, 0) => "db",
                _ => "ib",
            };
            let wb = if (inst >> 21) & 1 == 1 { ".wb" } else { "" };
            format!("{mnemonic}{mode}{wb}")
        }
        // Branch: 101
        0b101 => {
            let cond = inst >> 28;
            let suffix = match cond {
                0x0 => ".eq",
                0x1 => ".ne",
                0xE => "",
                _ => ".c?",
            };
            if (inst >> 24) & 1 == 1 {
                format!("bl{suffix}")
            } else {
                format!("b{suffix}")
            }
        }
        _ => format!("unknown.f{field20:02X}.p{:02X}", (inst >> 4) & 0xFF),
    }
}

thread_local! {
    /// Coverage collection: instruction word -> execution count (filled by UC_HOOK_CODE).
    static EXECUTED: std::cell::RefCell<HashMap<u32, u64>> =
        std::cell::RefCell::new(HashMap::new());
}

unsafe extern "C" fn coverage_code_hook(
    _engine: *mut std::os::raw::c_void,
    address: u64,
    _size: u32,
    user_data: *mut c_void,
) {
    // user_data = pointer to the current case's program slice
    let program = unsafe { &*(user_data as *const Vec<u32>) };
    let word = if address >= CODE as u64 && address < CODE as u64 + program.len() as u64 * 4 {
        program[((address - CODE as u64) / 4) as usize]
    } else {
        0 // fetch outside the program area (0 = andeq r0, r0)
    };
    EXECUTED.with(|e| *e.borrow_mut().entry(word).or_insert(0) += 1);
}

/// D2: long-run coverage statistics. `UC_COV_CASES` controls the case count (default 2000).
/// Produces `docs/COVERAGE.md`: families actually executed x hit counts + blind spots never executed.
#[test]
#[ignore = "D2 long-run tool: produces docs/COVERAGE.md"]
fn coverage_report() {
    let cov_cases = std::env::var("UC_COV_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let mut decoded: HashMap<String, u64> = HashMap::new();
    EXECUTED.with(|e| e.borrow_mut().clear());

    for case in 0..cov_cases {
        let seed = 0x5EED_0000 + case as u64;
        let mut rng = Rng(seed);
        let program = build_program(&mut rng);
        for word in &program {
            *decoded.entry(classify(*word)).or_insert(0) += 1;
        }

        let mut engine = std::ptr::null_mut();
        assert_eq!(
            unsafe { ffi::uc_open(ffi::UC_ARCH_ARM, ffi::UC_MODE_ARM, &mut engine) },
            0
        );
        unsafe {
            assert_eq!(ffi::uc_mem_map(engine, 0, MEMORY_SIZE, 7), 0);
            assert_eq!(ffi::uc_mem_map(engine, 0xFFFF_F000, 0x1000, 7), 0);
            let bytes = program_to_bytes(&program);
            assert_eq!(
                ffi::uc_mem_write(engine, CODE as u64, bytes.as_ptr(), bytes.len()),
                0
            );
            let mut init_rng = Rng(seed);
            for n in 0..13u32 {
                let value = DATA_START + ((init_rng.next() % 0x1FFF) as u32 & !3);
                assert_eq!(
                    ffi::uc_reg_write(engine, ffi::reg(n), &value as *const u32 as *const _),
                    0
                );
            }
            let cpsr = 0x13u32;
            assert_eq!(
                ffi::uc_reg_write(
                    engine,
                    ffi::UC_ARM_REG_CPSR,
                    &cpsr as *const u32 as *const _
                ),
                0
            );
            let pc0 = CODE;
            assert_eq!(
                ffi::uc_reg_write(engine, ffi::UC_ARM_REG_PC, &pc0 as *const u32 as *const _),
                0
            );
            let mut hook = 0usize;
            assert_eq!(
                ffi::uc_hook_add_code(
                    engine,
                    &mut hook,
                    ffi::UC_HOOK_CODE,
                    coverage_code_hook,
                    &program as *const Vec<u32> as *mut c_void,
                    1,
                    0,
                ),
                0
            );
            // Drive one instruction at a time (same cadence as the differential test; coverage only cares about what executed)
            for _ in 0..STEPS {
                let mut pc = 0u32;
                ffi::uc_reg_read(engine, ffi::UC_ARM_REG_PC, &mut pc as *mut u32 as *mut _);
                let begin = pc as u64;
                let err = ffi::uc_emu_start(engine, begin, begin + 4, 0, 1);
                if err != 0 {
                    break;
                }
            }
            ffi::uc_close(engine);
        }
    }

    // Aggregate: classification -> count
    let mut executed: HashMap<String, u64> = HashMap::new();
    EXECUTED.with(|e| {
        for (word, count) in e.borrow().iter() {
            *executed.entry(classify(*word)).or_insert(0) += count;
        }
    });

    let mut keys: Vec<&String> = decoded.keys().chain(executed.keys()).collect();
    keys.sort();
    keys.dedup();

    let mut report = String::new();
    report.push_str("# D2 Execution Coverage Report\n\n");
    report.push_str(&format!(
        "- Generation: {cov_cases} cases x {STEPS} instructions (seed base 0x5EED_0000); granularity in `tests/unicorn_diff.rs::classify`\n"
    ));
    report.push_str("- Generated: 2026-11-04\n\n");
    let total_exec: u64 = executed.values().sum();
    report.push_str(&format!(
        "- Total instructions actually executed: {total_exec}\n\n"
    ));
    report
        .push_str("| Family | Generated (decode) | Executed | Hit rate |\n|---|---:|---:|---:|\n");
    let mut misses = Vec::new();
    for key in &keys {
        let d = decoded.get(*key).copied().unwrap_or(0);
        let e = executed.get(*key).copied().unwrap_or(0);
        if e == 0 {
            misses.push((*key).clone());
        }
        let rate = if d == 0 {
            "n/a".to_string()
        } else {
            format!("{:.0}%", 100.0 * e as f64 / d as f64)
        };
        report.push_str(&format!("| `{key}` | {d} | {e} | {rate} |\n"));
    }
    if !misses.is_empty() {
        report.push_str(&format!(
            "\n## Families generated but never executed ({})\n\n",
            misses.len()
        ));
        for key in &misses {
            report.push_str(&format!("- `{key}`\n"));
        }
    }
    std::fs::create_dir_all("../docs").ok();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/COVERAGE.md");
    std::fs::write(path, &report).unwrap();
    println!(
        "coverage report written to {path} ({} families, {} blind spots)",
        keys.len(),
        misses.len()
    );
}

/// Triage helper: dump 200 random programs in order (PROGRAM_LEN words each) so
/// objdump can decode them offline and locate UNPREDICTABLE generator encodings.
#[test]
#[ignore = "triage tool"]
fn dump_programs_for_triage() {
    let mut out = Vec::new();
    let cases = cases();
    for case in 0..cases {
        let seed = 0x5EED_0000 + case as u64;
        let mut rng = Rng(seed);
        let program = build_program(&mut rng);
        assert_eq!(program.len(), PROGRAM_LEN);
        for word in program {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    let path = std::env::var("UC_PROGRAMS_BIN").unwrap_or_else(|_| "uc_programs.bin".into());
    std::fs::write(&path, &out).unwrap();
    println!("wrote {path} ({} bytes)", out.len());
}

/// D1 main test: compare random programs against unicorn (QEMU semantics).
/// Harness notes:
/// - On the unicorn side UC_HOOK_MEM_UNMAPPED maps 4 KB zero pages on demand, and
///   the top page of the 4 GB space is pre-mapped (mapping it mid-run returns rc=0
///   but has no effect, as observed with a probe); our SparseMemory has the same semantics.
/// - Every generator encoding was calibrated against an assembler and objdump
///   (bit25 in the DP immediate base, SMULxy Rd in bits 19-16, x/y halfword bits 5/6).
#[test]
fn unicorn_agrees_with_rynarmic_on_random_programs() {
    let mut state_mismatches = 0usize;
    let mut unicorn_errors = 0usize;
    let cases = cases();
    for case in 0..cases {
        let seed = 0x5EED_0000 + case as u64;
        let mut rng = Rng(seed);
        let program = build_program(&mut rng);

        let cpu = rynarmic_run(&program, seed);
        let (uc, uc_regs, uc_cpsr, uc_lr) = match unicorn_run(&program, seed) {
            Ok(result) => result,
            Err((err, pc, actual)) => {
                unicorn_errors += 1;
                let index = (pc.wrapping_sub(CODE) / 4) as usize;
                println!(
                    "case {case}: error {err}, pc={pc:#x} (instruction {index}) original={:08X?} actual={actual:08X}",
                    program.get(index).copied()
                );
                continue;
            }
        };

        let mut mismatch = Vec::new();
        for (n, (mine, uc)) in cpu.state.r.iter().zip(uc_regs.iter()).enumerate() {
            if mine != uc {
                mismatch.push(format!("r{n}: ours {mine:08x} vs uc {uc:08x}"));
            }
        }
        // r14 (LR) is also modified by BL, so compare it too
        if cpu.state.r[14] != uc_lr {
            mismatch.push(format!(
                "r14: ours {:08x} vs uc {:08x}",
                cpu.state.r[14], uc_lr
            ));
        }
        // Compare NZCV + Q (mode bits and E/J bits are outside this model's scope)
        let mask = 0xF800_0000u32;
        let mine_c = cpu.state.cpsr.bits() & mask;
        let uc_c = uc_cpsr & mask;
        if mine_c != uc_c {
            mismatch.push(format!("cpsr: ours {mine_c:08x} vs uc {uc_c:08x}"));
        }
        let mine_mem = memory_digest(&cpu);
        let uc_mem = uc.memory_digest();
        if mine_mem != uc_mem {
            mismatch.push(format!(
                "memory digest: ours {mine_mem:016x} vs uc {uc_mem:016x}"
            ));
        }
        if !mismatch.is_empty() {
            state_mismatches += 1;
            if state_mismatches <= 3 {
                println!(
                    "case {case}: {}\nprogram: {program:X?}",
                    mismatch.join("; ")
                );
            }
        }
    }
    println!(
        "unicorn differential: {cases} cases, {} compared, unicorn errors {unicorn_errors}, state mismatches {state_mismatches}",
        cases - unicorn_errors
    );
    assert_eq!(unicorn_errors, 0, "unicorn reported errors (see above)");
    assert_eq!(
        state_mismatches, 0,
        "semantic divergences from unicorn (see above)"
    );
}

/// Hand-written aligned smoke test: arithmetic + conditionals + aligned accesses +
/// SMUL, with no unaligned access anywhere, to verify the differential pipeline
/// end to end (generated cases are almost all skipped due to unaligned access; see above).
#[test]
fn unicorn_agrees_on_aligned_smoke() {
    let program: Vec<u32> = vec![
        0xE3A0_0064, // mov r0, #100
        0xE3A0_1003, // mov r1, #3
        0xE080_2001, // add r2, r0, r1
        0xE041_3002, // sub r3, r1, r2
        0xE3A0_10C8, // mov r1, #200
        0xE581_0000, // str r0, [r1]
        0xE591_4000, // ldr r4, [r1]
        0xE160_0081, // smulbb r0, r1, r1
        0xE0C2_5003, // sbc r5, r2, r3
        0xE354_0000, // cmp r4, #0
        0x13A0_6001, // movne r6, #1
        0xE3A0_702A, // mov r7, #42
        0xE087_7007, // add r7, r7, r7
    ];

    let cpu = rynarmic_run(&program, 1);
    let (uc, uc_regs, uc_cpsr, uc_lr) = match unicorn_run(&program, 1) {
        Ok(result) => result,
        Err((err, pc, _)) => panic!("unicorn smoke run failed: error {err}, PC={pc:#x}"),
    };

    for (n, (mine, uc)) in cpu.state.r.iter().zip(uc_regs.iter()).enumerate() {
        assert_eq!(mine, uc, "r{n} disagrees: ours {mine:08x} vs uc {uc:08x}");
    }
    assert_eq!(cpu.state.r[14], uc_lr, "r14 disagrees");
    let mask = 0xF800_0000u32;
    assert_eq!(
        cpu.state.cpsr.bits() & mask,
        uc_cpsr & mask,
        "CPSR(NZCV+Q) disagrees"
    );
    let _ = uc.memory_digest(); // Pipeline-availability placeholder; memory comparison happens in the generated cases
}

fn memory_digest(cpu: &Cpu<SparseMemory>) -> u64 {
    (DATA_START..0x8000u32).step_by(4).fold(0u64, |acc, addr| {
        acc.rotate_left(5)
            .wrapping_add(cpu.memory.peek_u32(addr) as u64)
    })
}

/// Probe: do the three instruction classes that write PC as data update the T bit
/// according to the ARMv5TE interworking semantics?
///
/// Triggered by a function return in real Thumb code (`pop {pc}`): bit0 of the return
/// address is the state bit; writing it to PC must **clear bit0 and set T from it**,
/// otherwise subsequent fetches use the wrong instruction set state.
#[test]
#[ignore = "triage probe: record unicorn interworking semantics for PC writes"]
fn probe_pc_interwork() {
    // (name, instruction word, is Thumb, value loaded into PC, register holding that value)
    let cases: [(&str, u32, bool, u32, u8); 6] = [
        ("arm  ldr pc,[r1] odd ", 0xE591_F000, false, 0x0000_3001, 0),
        ("arm  ldr pc,[r1] even", 0xE591_F000, false, 0x0000_3000, 0),
        ("arm  mov pc,r2   odd ", 0xE1A0_F002, false, 0x0000_3001, 2),
        ("arm  mov pc,r2   even", 0xE1A0_F002, false, 0x0000_3000, 2),
        // Thumb: pop {pc} (0xBD00). The low halfword comes first; the return value is
        // at the **top of the stack**.
        // (This unicorn FFI harness is opened with UC_MODE_ARM, so writing CPSR.T does
        //  not make it fetch in Thumb; the unicorn column below is therefore meaningless
        //  for Thumb cases. Thumb interworking semantics are covered by the ARM ARM
        //  plus the host-side m1/m4 end-to-end tests.)
        ("thumb pop {pc}   odd ", 0xBF00_BD00, true, 0x0000_3001, 14),
        ("thumb pop {pc}   even", 0xBF00_BD00, true, 0x0000_3000, 14),
    ];

    for (name, word, thumb, value, holder) in cases {
        let program = vec![word];
        let mut regs = [0u32; 13];
        regs[1] = DATA_START;
        let cpsr = 0x13 | if thumb { 0x20 } else { 0 };
        let mut uc = Unicorn::new(&program, &regs, cpsr).unwrap();
        let sp = DATA_START + 0x100;
        unsafe {
            if holder != 0 {
                let id = match holder {
                    14 => ffi::UC_ARM_REG_LR,
                    n => ffi::reg(n as u32),
                };
                assert_eq!(
                    ffi::uc_reg_write(uc.engine, id, &value as *const u32 as *const _),
                    0
                );
            }
            assert_eq!(
                ffi::uc_reg_write(uc.engine, 12, &sp as *const u32 as *const _),
                0
            );
            assert_eq!(
                ffi::uc_mem_write(
                    uc.engine,
                    DATA_START as u64,
                    &value as *const u32 as *const _,
                    4
                ),
                0
            );
        }
        let err = uc.run(1);

        let mut uc_pc = 0u32;
        let mut uc_cpsr = 0u32;
        unsafe {
            ffi::uc_reg_read(
                uc.engine,
                ffi::UC_ARM_REG_PC,
                &mut uc_pc as *mut u32 as *mut _,
            );
            ffi::uc_reg_read(
                uc.engine,
                ffi::UC_ARM_REG_CPSR,
                &mut uc_cpsr as *mut u32 as *mut _,
            );
        }

        let mut memory = SparseMemory::new();
        memory.write_words(CODE, &program);
        memory.write_words(DATA_START, &[value]);
        // `pop {pc}` reads from the top of the stack, not from lr.
        memory.write_words(sp, &[value]);
        let mut cpu = Cpu::new(memory);
        cpu.state.set_pc(CODE);
        cpu.state.set_reg(1, DATA_START);
        cpu.state.set_reg(2, value);
        cpu.state.set_reg(13, sp);
        cpu.state.set_reg(14, value);
        cpu.state.cpsr = Cpsr::from_bits(cpsr);
        let _ = cpu.step();
        let my_pc = cpu.state.pc();
        let my_t = cpu.state.cpsr.t();

        eprintln!(
            "{name}: value={value:08x} err={err}\n  unicorn pc={uc_pc:08x} T={} ours pc={my_pc:08x} T={my_t}",
            if uc_cpsr & 0x20 != 0 { 1 } else { 0 }
        );
    }
}
