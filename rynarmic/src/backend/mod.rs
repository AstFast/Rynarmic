//! Backend abstraction.
//!
//! Today there is only x86-64 (`dynasm-rs`). A backend compiles a
//! [`BasicBlock`](crate::BasicBlock) to machine code; any block it cannot compile stays with the
//! interpreter, so a backend is always allowed to cover only part of the instruction set.

// The code-memory container depends on std alone, so it compiles on every platform: it has to
// exist off x86-64 too, because `Core` holds one unconditionally. No block is ever compiled into
// it there.
pub mod code_arena;
#[cfg(target_arch = "x86_64")]
pub mod x64;

pub use code_arena::{CodeArena, MAX_ARENA_BYTES, MIN_BLOCK_ROOM};
#[cfg(target_arch = "x86_64")]
pub use x64::{ExchangeSlot, JitBlock, JitFn, compile};

/// The exchange-cache slot as it appears off x86-64: never constructed, and no machine code
/// refers to it.
#[cfg(not(target_arch = "x86_64"))]
#[derive(Default)]
pub struct ExchangeSlot;

#[cfg(not(target_arch = "x86_64"))]
impl ExchangeSlot {
    /// Placeholder: with no JIT, no machine code can fill this slot.
    pub fn store(&self, _pc: u32, _entry: *const u8) {}
}

#[cfg(not(target_arch = "x86_64"))]
mod stub {
    //! The placeholder backend for platforms other than x86-64: compilation is always declined and
    //! everything runs on the interpreter.
    //!
    //! `Cpu::register_compiled` type-checks on every platform, so this module's `JitBlock` must
    //! offer the whole set of query methods with the same names and shapes as the x64 backend —
    //! even though none of them is ever called, because `compile` always answers `None` and the
    //! driver never gets a `JitBlock` to hold.

    use crate::block::BasicBlock;
    use crate::memory::{FastWindow, MAX_WINDOWS, Memory};
    use crate::state::ArmState;
    use std::sync::atomic::AtomicU64;

    /// A link slot placeholder: `slots()` is always empty, and these fields exist only so that the
    /// construction of `LinkSite` still type-checks.
    pub struct LinkSlot {
        pub stub: u64,
        pub target_pc: u32,
        pub thumb: bool,
    }

    impl LinkSlot {
        pub fn cell_ptr(&self) -> *const AtomicU64 {
            std::ptr::null()
        }
    }

    /// A placeholder type that is never constructed.
    pub struct JitBlock;

    /// The placeholder signature, kept in the same shape as the real backend's.
    pub type JitFn = unsafe extern "C" fn(*mut ArmState, *mut u8) -> u32;

    impl JitBlock {
        pub fn code_len(&self) -> usize {
            0
        }

        pub fn exchange_ptr(&self) -> *const super::ExchangeSlot {
            std::ptr::null()
        }

        pub fn entry_ptr(&self) -> *const u8 {
            std::ptr::null()
        }

        pub fn body_ptr(&self) -> *const u8 {
            std::ptr::null()
        }

        pub fn start_pc(&self) -> u32 {
            0
        }

        pub fn thumb(&self) -> bool {
            false
        }

        pub fn needs_memory(&self) -> bool {
            false
        }

        pub fn slots(&self) -> impl Iterator<Item = &LinkSlot> {
            std::iter::empty()
        }

        /// # Safety
        ///
        /// The placeholder implementation does nothing.
        pub unsafe fn run<M>(&self, _state: *mut ArmState, _memory: *mut M) -> u32 {
            0
        }
    }

    /// Off x86-64 there is no backend; the window argument is there only to keep the signature in
    /// step with the real `compile`.
    pub fn compile<M: Memory>(
        _block: &BasicBlock,
        _thumb: bool,
        _windows: [Option<FastWindow>; MAX_WINDOWS],
    ) -> Option<JitBlock> {
        None
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub use stub::{JitBlock, JitFn, compile};
