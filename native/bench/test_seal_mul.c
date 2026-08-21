// Test: LLVM schoolbook recognition with SEAL's exact pattern
//
// The key insight: multiply_uint64_generic computes BOTH low and high 64 bits
// (result128[0] = low, result128[1] = high). This is a "full 128-bit multiply"
// pattern that LLVM can recognize as a single `mul` on AMD64.
//
// My earlier mulhi_schoolbook only computed the HIGH half — that's a partial
// pattern that LLVM might NOT recognize.
//
// Build:
//   clang -O3 -S -o test_seal_mul_clang.s test_seal_mul.c
//   gcc -O3 -S -o test_seal_mul_gcc.s test_seal_mul.c
//
// Check:
//   sed -n '/multiply_uint64_generic:/,/ret/p' test_seal_mul_clang.s | grep -c "mul"
//   # LLVM: expect 1 (recognized as mul)
//   # GCC:  expect 0 (schoolbook, no single mul)

#include <stdint.h>
#include <stdio.h>

// Exact copy of SEAL's multiply_uint64_generic (from uintarith.h)
inline unsigned char add_uint64(uint64_t operand1, uint64_t operand2,
                                unsigned long long *result) {
  *result = operand1 + operand2;
  return static_cast<unsigned char>(*result < operand1);
}

__attribute__((noinline))
void multiply_uint64_generic(uint64_t operand1, uint64_t operand2,
                             unsigned long long *result128) {
  auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
  auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
  operand1 >>= 32;
  operand2 >>= 32;

  auto middle1 = operand1 * operand2_coeff_right;
  unsigned long long middle;
  auto left = operand1 * operand2 +
              (static_cast<uint64_t>(add_uint64(
                   middle1, operand2 * operand1_coeff_right, &middle))
               << 32);
  auto right = operand1_coeff_right * operand2_coeff_right;
  auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

  result128[1] =
      static_cast<unsigned long long>(left + (middle >> 32) + (temp_sum >> 32));
  result128[0] = static_cast<unsigned long long>((temp_sum << 32) |
                                                 (right & 0x00000000FFFFFFFF));
}

// Also test: only-high variant (what my earlier test used)
__attribute__((noinline))
uint64_t mulhi_only(uint64_t a, uint64_t b) {
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

// Reference: __uint128_t
__attribute__((noinline))
void multiply_uint128(uint64_t a, uint64_t b, unsigned long long *result128) {
    unsigned __int128 prod = (unsigned __int128)a * b;
    result128[0] = (unsigned long long)prod;
    result128[1] = (unsigned long long)(prod >> 64);
}

int main() {
    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;
    unsigned long long r1[2], r2[2];

    multiply_uint64_generic(a, b, r1);
    multiply_uint128(a, b, r2);

    printf("generic:  lo=%016lx hi=%016lx\n", r1[0], r1[1]);
    printf("uint128:  lo=%016lx hi=%016lx\n", r2[0], r2[1]);
    printf("match:    %s\n", (r1[0] == r2[0] && r1[1] == r2[1]) ? "YES" : "NO");

    uint64_t hi = mulhi_only(a, b);
    printf("mulhi_only: hi=%016lx (should be %016lx)\n", hi, r2[1]);

    return 0;
}
