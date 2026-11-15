//! A minimal unicorn FFI harness: map, write `mov r0,#42; swi`, run, read r0.
//!
//! Compiles to a no-op stub unless build.rs enables `unicorn_ref` (see
//! `RYNARMIC_UNICORN`), so a clean checkout still builds every example.

#[cfg(unicorn_ref)]
mod ffi {
    use std::os::raw::{c_int, c_void};
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
        pub fn uc_reg_read(engine: *mut c_void, regid: c_int, value: *mut c_void) -> c_int;
        pub fn uc_reg_write(engine: *mut c_void, regid: c_int, value: *const c_void) -> c_int;
        pub fn uc_emu_start(
            engine: *mut c_void,
            begin: u64,
            until: u64,
            timeout: u64,
            count: usize,
        ) -> c_int;
    }
}

#[cfg(unicorn_ref)]
fn main() {
    // Reference rynarmic so that this package's build-script link flags propagate to this example
    let _ = rynarmic::Cpsr::from_bits(0);
    let mut engine = std::ptr::null_mut();
    let err = unsafe { ffi::uc_open(1, 0, &mut engine) };
    println!("uc_open: {err}");
    let err = unsafe { ffi::uc_mem_map(engine, 0x1000, 0x1000, 7) }; // 7 = UC_PROT_ALL
    println!("uc_mem_map: {err}");
    let code: [u8; 8] = [
        0x2C, 0x00, 0xA0, 0xE3, // mov r0, #42
        0x00, 0x00, 0x00, 0xEF, // swi 0
    ];
    let err = unsafe { ffi::uc_mem_write(engine, 0x1000, code.as_ptr(), code.len()) };
    println!("uc_mem_write: {err}");
    let cpsr = 0x13u32;
    unsafe { ffi::uc_reg_write(engine, 3, &cpsr as *const u32 as *const _) };
    let err = unsafe { ffi::uc_emu_start(engine, 0x1000, 0x1008, 0, 10) };
    println!("uc_emu_start: {err}");
    let mut r0 = 0u32;
    unsafe { ffi::uc_reg_read(engine, 66, &mut r0 as *mut u32 as *mut _) };
    println!("r0 = {r0} (expected 42)");
    unsafe { ffi::uc_close(engine) };
}

#[cfg(not(unicorn_ref))]
fn main() {
    println!(
        "unicorn smoke test disabled: set RYNARMIC_UNICORN to enable the differential reference"
    );
}
