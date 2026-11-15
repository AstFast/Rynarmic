//! Checks that dynasm-rs works here: reserve executable memory, generate code, call it.
//!
//! Usage: `cargo run --example jit_smoke`

use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm, x64::Assembler};

fn main() {
    let mut asm = Assembler::new().expect("could not reserve executable memory");

    // extern "win64" fn(u64) -> u64: the argument in RCX, the result in RAX
    dynasm!(asm
        ; .arch x64
        ; mov rax, rcx
        ; add rax, 1
        ; ret
    );

    let buf = asm.finalize().expect("finalize failed");
    let entry = buf.ptr(dynasmrt::AssemblyOffset(0));
    let f: extern "win64" fn(u64) -> u64 = unsafe { std::mem::transmute(entry) };

    let value = f(41);
    println!("generated function f(41) = {value}");
    assert_eq!(value, 42, "the generated code returned the wrong value");

    // Labels and jumps: confirm the relative displacement is backpatched correctly at run time
    let mut asm = Assembler::new().expect("could not reserve executable memory");
    dynasm!(asm
        ; .arch x64
        ; xor eax, eax
        ; jmp >done
        ; mov eax, 0xDEAD
        ; done:
        ; ret
    );
    let buf = asm.finalize().expect("finalize failed");
    let entry = buf.ptr(dynasmrt::AssemblyOffset(0));
    let f: extern "win64" fn() -> u64 = unsafe { std::mem::transmute(entry) };
    assert_eq!(f(), 0, "the label jump did not take effect");

    println!(
        "dynasm-rs works: executable memory, argument passing and label backpatching are all fine"
    );
}
