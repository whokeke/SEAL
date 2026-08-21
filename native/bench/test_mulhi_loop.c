#include <stdint.h>
#include <stdio.h>
#include <sched.h>
#include <string.h>

inline unsigned char add_uint64(uint64_t a, uint64_t b, uint64_t *result) {
    *result = a + b;
    return (unsigned char)(*result < a);
}

// 版本1: 手动 carry（LLVM 不识别 → schoolbook）
static inline uint64_t mulhi_manual_carry(uint64_t a, uint64_t b) {
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

// 版本2: add_uint64 carry（LLVM 识别 → umulh）
static inline uint64_t mulhi_add_uint64(uint64_t a, uint64_t b) {
    uint64_t a_lo = a & 0xFFFFFFFF, a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF, b_hi = b >> 32;
    uint64_t p_ll = a_lo * b_lo, p_hh = a_hi * b_hi;
    uint64_t p_hl = a_hi * b_lo, p_lh = a_lo * b_hi;
    uint64_t mid;
    uint64_t carry1 = add_uint64(p_hl, p_lh, &mid);
    uint64_t left = p_hh + (carry1 << 32);
    uint64_t temp_sum = (p_ll >> 32) + (mid & 0xFFFFFFFF);
    return left + (mid >> 32) + (temp_sum >> 32);
}

// 版本3: __uint128_t 参考
static inline uint64_t mulhi_u128(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b >> 64);
}

#define N (1 << 16)
#define ITERS 10000

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }

// volatile 防止 DCE
volatile uint64_t sink;

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    static uint64_t in1[N], in2[N];
    // 用大数确保 mulhi 结果非零
    for (int i = 0; i < N; i++) { in1[i] = 0x123456789ABCDEF0ULL * (i+1); in2[i] = 0xFEDCBA9876543210ULL * (i+1); }

    printf("=== mulhi loop benchmark (N=%d, ITERS=%d) ===\n\n", N, ITERS);

    uint64_t t0, t1, acc;

    // manual carry
    acc = 0;
    t0 = cntvct();
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < N; i++)
            acc ^= mulhi_manual_carry(in1[i], in2[i]);
    t1 = cntvct();
    sink = acc;
    printf("  manual_carry  %7.3f cyc/elem  (sink=%lx)\n", (double)(t1-t0)*23.0/(ITERS*N), (unsigned long)acc);

    // add_uint64 carry
    acc = 0;
    t0 = cntvct();
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < N; i++)
            acc ^= mulhi_add_uint64(in1[i], in2[i]);
    t1 = cntvct();
    sink = acc;
    printf("  add_uint64    %7.3f cyc/elem  (sink=%lx)\n", (double)(t1-t0)*23.0/(ITERS*N), (unsigned long)acc);

    // __uint128_t
    acc = 0;
    t0 = cntvct();
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < N; i++)
            acc ^= mulhi_u128(in1[i], in2[i]);
    t1 = cntvct();
    sink = acc;
    printf("  __uint128_t   %7.3f cyc/elem  (sink=%lx)\n", (double)(t1-t0)*23.0/(ITERS*N), (unsigned long)acc);

    // verify
    uint64_t r0 = mulhi_manual_carry(in1[0], in2[0]);
    uint64_t r1 = mulhi_add_uint64(in1[0], in2[0]);
    uint64_t r2 = mulhi_u128(in1[0], in2[0]);
    printf("\n  verify: manual=%016lx add_u64=%016lx u128=%016lx %s\n",
        (unsigned long)r0, (unsigned long)r1, (unsigned long)r2,
        (r0==r1 && r1==r2) ? "MATCH" : "MISMATCH");

    return 0;
}
