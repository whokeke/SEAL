// Test: Can LLVM 22 recognize schoolbook mulhi as a single mul instruction?
//
// Build with GCC:
//   gcc -O3 -o test_mulhi_gcc test_mulhi.c
//   gcc -O3 -S -o test_mulhi_gcc.s test_mulhi.c
//
// Build with LLVM:
//   clang -O3 -o test_mulhi_clang test_mulhi.c
//   clang -O3 -S -o test_mulhi_clang.s test_mulhi.c
//
// Check assembly:
//   grep -c "mul\|umulh" test_mulhi_gcc.s    # GCC: expect 0 (schoolbook, no mul)
//   grep -c "mul\|umulh" test_mulhi_clang.s  # LLVM: expect 1 (recognized as mul)
//
// Run:
//   ./test_mulhi_gcc
//   ./test_mulhi_clang
//   # Both should print same result

#include <stdio.h>
#include <stdint.h>

// Schoolbook: 64x64->high64 via 32-bit decomposition
// This is the pattern LLVM 22 PR #168396 recognizes on AMD64
__attribute__((noinline))
uint64_t mulhi_schoolbook(uint64_t a, uint64_t b) {
    uint64_t a_lo = a & 0xFFFFFFFF, a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF, b_hi = b >> 32;
    uint64_t p_ll = a_lo * b_lo, p_hh = a_hi * b_hi;
    uint64_t p_hl = a_hi * b_lo, p_lh = a_lo * b_hi;
    uint64_t mid = p_hl + p_lh;
    uint64_t carry1 = (mid < p_hl);
    uint64_t mid_lo = mid & 0xFFFFFFFF, mid_hi = mid >> 32;
    uint64_t t = p_ll + (mid_lo << 32);
    uint64_t carry2 = (t < p_ll);
    return p_hh + mid_hi + (carry1 << 32) + carry2;
}

// Reference: __uint128_t (generates mul on AMD64, mul+umulh on ARM64)
__attribute__((noinline))
uint64_t mulhi_u128(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b >> 64);
}

int main() {
    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;

    uint64_t r1 = mulhi_schoolbook(a, b);
    uint64_t r2 = mulhi_u128(a, b);

    printf("schoolbook: 0x%016lx\n", (unsigned long)r1);
    printf("__uint128_t: 0x%016lx\n", (unsigned long)r2);
    printf("match: %s\n", r1 == r2 ? "YES" : "NO");

    // Also test with loop to make it easier to benchmark
    uint64_t acc = 0;
    for (int i = 0; i < 100000000; i++) {
        acc ^= mulhi_schoolbook(a + i, b + i);
    }
    printf("loop result: 0x%016lx\n", (unsigned long)acc);

    return 0;
}
