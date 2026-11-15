/* Benchmark kernels that are really compiled (arm-none-eabi-gcc, ARMv5TEJ).
 *
 * No libc and no OS: pure computation only; the return value is the checksum.
 * The three kernels mirror the three most common shapes in real code:
 *   0 = division plus branches (calls libgcc's __aeabi_uidiv)
 *   1 = word copy plus accumulation (memory-bound)
 *   2 = repeated calls to a small function (call-bound)
 *
 * The expected values are recomputed in Rust by tests/compiled_workload.rs and
 * compared, so a mistake here -- or in the emulator -- is caught by that test.
 */

typedef unsigned int u32;

__attribute__((noinline)) static u32 kern_div(u32 rounds, u32 seed) {
    u32 acc = seed;
    for (u32 i = 1; i <= rounds; i++) {
        acc += 0xFFFF0000u / (i | 1);
        acc ^= acc >> 3;
        if (acc & 1) {
            acc += 0x9E3779B9u;
        } else {
            acc -= 0x7F4A7C15u;
        }
    }
    return acc;
}

__attribute__((noinline)) static u32 kern_copy(u32 *dst, const u32 *src, u32 words) {
    u32 sum = 0;
    for (u32 i = 0; i < words; i++) {
        dst[i] = src[i] + i;
        sum += dst[i];
    }
    return sum;
}

__attribute__((noinline)) static u32 step(u32 x) {
    return (x << 1) ^ (x >> 3) ^ 0x1234u;
}

__attribute__((noinline)) static u32 kern_call(u32 rounds, u32 seed) {
    u32 acc = seed;
    for (u32 i = 0; i < rounds; i++) {
        acc = step(acc + i);
    }
    return acc;
}

u32 bench_main(u32 which, u32 a, u32 b, u32 c) {
    switch (which) {
    case 0:
        return kern_div(a, b);
    case 1:
        return kern_copy((u32 *)a, (const u32 *)b, c);
    default:
        return kern_call(a, b);
    }
}
