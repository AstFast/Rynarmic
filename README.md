# Rynarmic

A 32-bit ARM (ARMv5TEJ) execution core in Rust: interpreter, block cache and x86-64 JIT.

**This crate is a Rust rewrite of code from Dynarmic**, not a from-scratch implementation. Its `decoder -> ir -> executor/backend` layering, host memory callback contract and part of the instruction semantics come from that codebase.

**An AI agent rewrote the Dynarmic C++ sources into Rust. Human contribution was limited to testing.**

```sh
cargo build
```

Rust 1.88+ (edition 2024); no system libraries needed.

Optional experiment tooling — not build dependencies: `RYNARMIC_ARM_GCC` (performance samples) and `RYNARMIC_UNICORN` (differential tests). A fresh clone needs neither.

**Licence:** `MIT AND 0BSD` — [`LICENSE`](LICENSE) for the code written for this crate, [`LICENSE-DYNARMIC.txt`](LICENSE-DYNARMIC.txt) for the Dynarmic-derived code (0BSD, Copyright (C) 2017 merryhime).
