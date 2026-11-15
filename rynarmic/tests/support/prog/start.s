/* Entry stub: call the C entry point and, on return, hand control back to the emulator with SWI.
 *
 * SWI (rather than an infinite loop) is used because run_cached returns it as Err:
 * the caller learns that execution finished and also gets the exact instruction
 * count, so the spin loop's idle iterations are not counted in the throughput.
 */
    .syntax unified
    .arm
    .text
    .global _start
_start:
    bl bench_main
    swi #0
