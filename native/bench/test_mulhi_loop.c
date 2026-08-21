#include <stdint.h>
#include <stdio.h>
#include <sched.h>

inline unsigned char add_uint64(uint64_t a, uint64_t b, uint64_t *result) {
    *result = a + b;
    return (unsigned char)(*result < a);
}

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

static inline uint64_t mulhi_u128(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b >> 64);
}

#define N (1 << 16)
#define ITERS 10000

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }

volatile uint64_t sink;

// 每个函数包含完整循环，noinline 确保独立编译
__attribute__((noinline))
void run_manual_carry(const uint64_t *in1, const uint64_t *in2, uint64_t *out, int n) {
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < n; i++)
            out[i] = mulhi_manual_carry(in1[i], in2[i]);
}

__attribute__((noinline))
void run_add_uint64(const uint64_t *in1, const uint64_t *in2, uint64_t *out, int n) {
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < n; i++)
            out[i] = mulhi_add_uint64(in1[i], in2[i]);
}

__attribute__((noinline))
void run_u128(const uint64_t *in1, const uint64_t *in2, uint64_t *out, int n) {
    for (int iter = 0; iter < ITERS; iter++)
        for (int i = 0; i < n; i++)
            out[i] = mulhi_u128(in1[i], in2[i]);
}

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    static uint64_t in1[N], in2[N], out[N];
    for (int i = 0; i < N; i++) { in1[i] = 0x123456789ABCDEF0ULL * (i+1); in2[i] = 0xFEDCBA9876543210ULL * (i+1); }

    printf("=== mulhi loop benchmark (N=%d, ITERS=%d) ===\n\n", N, ITERS);

    uint64_t t0, t1;

    t0 = cntvct(); run_manual_carry(in1, in2, out, N); t1 = cntvct();
    sink = out[0] ^ out[N-1];
    printf("  manual_carry  %7.3f cyc/elem\n", (double)(t1-t0)*23.0/(ITERS*N));

    t0 = cntvct(); run_add_uint64(in1, in2, out, N); t1 = cntvct();
    sink = out[0] ^ out[N-1];
    printf("  add_uint64    %7.3f cyc/elem\n", (double)(t1-t0)*23.0/(ITERS*N));

    t0 = cntvct(); run_u128(in1, in2, out, N); t1 = cntvct();
    sink = out[0] ^ out[N-1];
    printf("  __uint128_t   %7.3f cyc/elem\n", (double)(t1-t0)*23.0/(ITERS*N));

    // verify
    uint64_t r0 = mulhi_manual_carry(in1[0], in2[0]);
    uint64_t r1 = mulhi_add_uint64(in1[0], in2[0]);
    uint64_t r2 = mulhi_u128(in1[0], in2[0]);
    printf("\n  verify: %s\n", (r0==r1 && r1==r2) ? "MATCH" : "MISMATCH");

    return 0;
}
