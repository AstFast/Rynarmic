//! Rynarmic — a dynamic recompiler for ARM A32 and Thumb (ARMv5TEJ).
//!
//! This crate incorporates code rewritten from Dynarmic (0BSD, Copyright (C) 2017
//! merryhime); see the repository README and LICENSE-DYNARMIC.txt for the full
//! attribution and licence text. An AI agent rewrote the Dynarmic C++ sources into
//! Rust; human contribution was limited to testing. The
//! `decoder -> ir -> executor/backend` layering follows Dynarmic's. Two execution
//! paths exist today, an interpreter and x64 (through dynasm-rs), and both share
//! one IR and one block cache.

pub mod backend;
pub mod block;
pub mod common;
pub mod cpu;
pub mod decoder;
pub mod executor;
pub mod ir;
pub mod memory;
pub mod state;
pub mod trace;

pub use block::BasicBlock;
pub use cpu::{Cpu, ExceptionKind, StepOutcome};
pub use decoder::DecodeError;
pub use ir::{AluOp, IrOp, MemSize, ShiftType, Shifter, Value};
pub use memory::Memory;
pub use state::{ArmState, Cpsr, Mode};
