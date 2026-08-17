// Benchmark: Lazy Barrett mulmod — NEON vs SVE vs scalar
// Tests what happens when only NEON (Advanced SIMD) is available,
// no SVE/SVE2. NEON has no umulh equivalent, so schoolbook is mandatory.
//
// Build with NEON only (no SVE):
//   gcc -O3 -march=armv8-a+crypto -D_GNU_SOURCE -o bench_mulmod_neon bench_mulmod_neon.c
// Build with SVE (for comparison):
//   gcc -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_mulmod_neon_sve bench_mulmod_neon.c
// Run: taskset -c 1 ./bench_mulmod_neon

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_neon.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   (1000)

static inline uint64_t cntvct(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v;
}

/* ============================================================
 * (1) SCALAR SCHOOLBOOK — 32-bit decomposition (~16 instructions)
 * ============================================================ */
static inline uint64_t mulhi_schoolbook(uint64_t a, uint64_t b) {
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

__attribute__((noinline))
static void scalar_schoolbook(const uint64_t *in, uint64_t *out,
                              uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (2) SCALAR __uint128_t — compiler generates umulh
 * ============================================================ */
__attribute__((noinline))
static void scalar_int128(const uint64_t *in, uint64_t *out,
                          uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        unsigned __int128 prod = (unsigned __int128)in[i] * q;
        uint64_t hi = (uint64_t)(prod >> 64);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* (3) NEON schoolbook moved below */

// Cleaner approach: NEON for parallel multiplies, scalar for carry
__attribute__((noinline))
static void neon_schoolbook(const uint64_t *in, uint64_t *out,
                            uint64_t op, uint64_t q, uint64_t p, int n) {
    // Process 2 elements at a time using NEON
    int i = 0;
    int full = n & ~1;
    for (; i < full; i += 2) {
        // Load 2 elements
        uint64x2_t va = vld1q_u64(in + i);
        uint64x2_t vq = vdupq_n_u64(q);

        // Split into 32-bit halves
        uint32x2_t a_lo = vmovn_u64(va);
        uint32x2_t a_hi = vshrn_n_u64(va, 32);
        uint32x2_t q_lo = vmovn_u64(vq);
        uint32x2_t q_hi = vshrn_n_u64(vq, 32);

        // 4 parallel 32x32->64 multiplies (2 lanes each)
        uint64x2_t p_ll = vmull_u32(a_lo, q_lo);
        uint64x2_t p_hh = vmull_u32(a_hi, q_hi);
        uint64x2_t p_hl = vmull_u32(a_hi, q_lo);
        uint64x2_t p_lh = vmull_u32(a_lo, q_hi);

        // Extract to scalar for carry chain (NEON has no good carry detection)
        uint64_t p_ll_0 = vgetq_lane_u64(p_ll, 0);
        uint64_t p_ll_1 = vgetq_lane_u64(p_ll, 1);
        uint64_t p_hh_0 = vgetq_lane_u64(p_hh, 0);
        uint64_t p_hh_1 = vgetq_lane_u64(p_hh, 1);
        uint64_t p_hl_0 = vgetq_lane_u64(p_hl, 0);
        uint64_t p_hl_1 = vgetq_lane_u64(p_hl, 1);
        uint64_t p_lh_0 = vgetq_lane_u64(p_lh, 0);
        uint64_t p_lh_1 = vgetq_lane_u64(p_lh, 1);

        // Lane 0: carry chain
        uint64_t mid0 = p_hl_0 + p_lh_0;
        uint64_t carry1_0 = (mid0 < p_hl_0);
        uint64_t mid_lo0 = mid0 & 0xFFFFFFFF;
        uint64_t mid_hi0 = mid0 >> 32;
        uint64_t t0 = p_ll_0 + (mid_lo0 << 32);
        uint64_t carry2_0 = (t0 < p_ll_0);
        uint64_t hi0 = p_hh_0 + mid_hi0 + (carry1_0 << 32) + carry2_0;

        // Lane 1: carry chain
        uint64_t mid1 = p_hl_1 + p_lh_1;
        uint64_t carry1_1 = (mid1 < p_hl_1);
        uint64_t mid_lo1 = mid1 & 0xFFFFFFFF;
        uint64_t mid_hi1 = mid1 >> 32;
        uint64_t t1 = p_ll_1 + (mid_lo1 << 32);
        uint64_t carry2_1 = (t1 < p_ll_1);
        uint64_t hi1 = p_hh_1 + mid_hi1 + (carry1_1 << 32) + carry2_1;

        // Finish mulmod for both lanes
        uint64_t lo0 = in[i] * op;
        uint64_t r0 = lo0 - hi0 * p;
        out[i] = (r0 >= p) ? r0 - p : r0;

        uint64_t lo1 = in[i+1] * op;
        uint64_t r1 = lo1 - hi1 * p;
        out[i+1] = (r1 >= p) ? r1 - p : r1;
    }
    // Tail
    for (; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (4) NEON __uint128_t — let compiler auto-vectorize with NEON
 * On AArch64 without SVE, compiler uses NEON for __uint128_t
 * but has no umulh instruction (only mul+umulh scalar).
 * Actually, AArch64 DOES have scalar umulh instruction.
 * The question is whether NEON auto-vectorization can use it.
 * Answer: No, NEON has no vector umulh. But scalar umulh works.
 * ============================================================ */
__attribute__((noinline))
static void neon_int128(const uint64_t *in, uint64_t *out,
                        uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        unsigned __int128 prod = (unsigned __int128)in[i] * q;
        uint64_t hi = (uint64_t)(prod >> 64);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================ */
int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    uint64_t *in  = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t *out = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t p = 0xFFFFFFFFFFFFFFC5ULL;
    for (int i = 0; i < N_ELEMS; i++) in[i] = (i * 123456789ULL + 1) % p;
    uint64_t op = 0x123456789ABCDEF0ULL % p;
    uint64_t q  = op;

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, int);
    struct { const char *name; fn_t fn; } tests[] = {
        {"scalar schoolbook",   scalar_schoolbook},
        {"scalar __uint128_t",   scalar_int128},
        {"neon schoolbook",      neon_schoolbook},
        {"neon __uint128_t",     neon_int128},
    };

    printf("=== NEON vs scalar mulmod benchmark ===\n");
    printf("    N=%d ITERS=%d VL=128bit(N=2) HIP12 2.3GHz\n\n", N_ELEMS, ITERS);

    double base = 0;
    for (int t = 0; t < 4; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t0 = cntvct();
        for (int j = 0; j < ITERS; j++)
            tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t1 = cntvct();
        double cyc = (double)(t1-t0) * 23.0 / (ITERS * N_ELEMS);
        if (t == 0) base = cyc;
        printf("    %-25s %7.3f cyc/elem  (%.2fx vs schoolbook)\n",
               tests[t].name, cyc, base/cyc);
    }

    /* correctness */
    uint64_t ref[N_ELEMS], v[N_ELEMS];
    scalar_int128(in, ref, op, q, p, N_ELEMS);
    int ok = 1;
    fn_t others[] = {scalar_schoolbook, neon_schoolbook, neon_int128};
    const char *names[] = {"schoolbook", "neon-schoolbook", "neon-int128"};
    for (int t = 0; t < 3; t++) {
        others[t](in, v, op, q, p, N_ELEMS);
        for (int i = 0; i < N_ELEMS; i++) {
            if (v[i] != ref[i]) { printf("    MISMATCH %s[%d]\n", names[t], i); ok=0; break; }
        }
    }
    printf("\n    Correctness: %s\n", ok ? "ALL MATCH" : "FAILED");
    free(in); free(out);
    return 0;
}
