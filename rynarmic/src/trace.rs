//! A per-instruction register trace over one guest address window, off unless asked for.
//!
//! `RYNARMIC_TRACE_PC=lo,hi` (hex, `0x` optional) turns it on. Two pieces cooperate, and both are
//! needed:
//!
//! - [`covers`] caps a translated block at **one instruction** while it is inside the window. A
//!   compiled block is a flat `Vec<IrOp>` and the mapping from an operation back to the guest
//!   address that produced it is gone by the time it runs (the optimizer rebuilds the vector), so a
//!   per-operation trace is not available at run time. Making the block boundary the instruction
//!   boundary buys the same thing without touching the IR: the block driver then stands at the start
//!   of exactly one instruction.
//! - `Cpu::run_interpreted` prints the register file at that point -- inside the window every block
//!   is one instruction, so this is one line per guest instruction. Printing from the block driver
//!   rather than from `Cpu::step` is what makes the trace work on the *translated* path as well:
//!   a one-instruction block is still compiled unless the window is what stopped it.
//!
//! Block boundaries carry no architectural meaning — the same instructions run with the same
//! meaning — so the trace is faithful rather than a re-interpretation. It does change how the
//! instructions are *grouped*, so a defect that depends on grouping can hide under it. Seeing the
//! trace go clean is therefore itself a finding: it moves the defect into the block-level layer.
//!
//! The window is read once. This test sits on every block translation and on every step, and
//! `std::env::var_os` takes a process-wide lock — the same reason `RYNARMIC_IR_DEBUG` is cached.

use std::sync::OnceLock;

/// The half-open guest address window `[lo, hi)` the trace applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub lo: u32,
    pub hi: u32,
}

fn parse(text: &str) -> Option<Window> {
    let (lo, hi) = text.split_once(',')?;
    let number = |s: &str| {
        let s = s.trim();
        match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => s.parse().ok(),
        }
    };
    let (lo, hi) = (number(lo)?, number(hi)?);
    (lo < hi).then_some(Window { lo, hi })
}

/// The window from the environment, if one was asked for.
pub fn window() -> Option<Window> {
    static WINDOW: OnceLock<Option<Window>> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("RYNARMIC_TRACE_PC")
            .ok()
            .and_then(|text| parse(&text))
    })
}

/// Whether `pc` falls inside the window, i.e. whether a block starting there has to hold exactly
/// one instruction.
///
/// Also answers false when no window was asked for, so the translator's cap is its normal value.
pub fn covers(pc: u32) -> bool {
    window().is_some_and(|w| w.lo <= pc && pc < w.hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_parses_hex_decimal_and_rejects_inverted_ranges() {
        assert!(parse("0x10000000,0x10000010").is_some());
        assert!(parse("0X10,20").is_some());
        assert!(parse("16,32").is_some());
        // hi <= lo names an empty window, and a missing comma is not a window
        assert_eq!(parse("0x20,0x20"), None);
        assert_eq!(parse("0x20,0x10"), None);
        assert_eq!(parse("0x20"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn covers_is_half_open_and_off_without_a_window() {
        // `covers` reads the process environment, which this test cannot set safely; the half-open
        // arithmetic is what the translator relies on, so it is checked through the same bounds the
        // parser produces.
        let w = parse("0x100,0x200").unwrap();
        let inside = |pc: u32| w.lo <= pc && pc < w.hi;
        assert!(inside(0x100) && inside(0x1ff));
        assert!(!inside(0xff) && !inside(0x200));
    }
}
