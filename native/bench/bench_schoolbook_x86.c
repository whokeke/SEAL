// Pure schoolbook benchmark (x86/AVX-512, no umulh)
// On x86 there is NO 64x64->high64 instruction — schoolbook is mandatory.
// __uint128_t generates `mul` (rdx:rax), which is also 1 instruction but
// has no vectorized form. This benchmark isolates schoolbook performance.
//
// Build: gcc -O3 -mavx512f -mavx512dq -D_GNU_SOURCE -o bench_schoolbook_x86 bench_schoolbook_x86.c
// Run:   taskset -c 1 ./bench_schoolbook_x86
//
// Variants:
//   (1) scalar schoolbook — 1 element/iter, no vectorization
//   (2) AVX-512 schoolbook — 8 elements/iter, _mm512_mul_epu32 (32x32->64, 8 lanes)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <sched.h>

/* Use rdtsc for x86 timing (this file is x86-only) */
static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#define N_ELEMS (65536)
#define ITERS   (1000)

/* ============================================================
 * (1) SCALAR SCHOOLBOOK — 32-bit decomposition, ~16 instructions
 * Prevent auto-vectorization to get true scalar performance
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

__attribute__((noinline,optimize("no-tree-vectorize")))
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
 * (2) AVX-512 SCHOOLBOOK — _mm512_mul_epu32 (32x32->64), 8 lanes
 * This is the ONLY vectorized option on x86 (no vector umulh).
 * SEAL's avx512_mulhi_epu64 is exactly this approach.
 * ============================================================ */
static inline __m512i avx512_mulhi_epu64(__m512i a, __m512i b) {
    __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFF);
    __m512i a_lo = _mm512_and_si512(a, mask32);
    __m512i a_hi = _mm512_srli_epi64(a, 32);
    __m512i b_lo = _mm512_and_si512(b, mask32);
    __m512i b_hi = _mm512_srli_epi64(b, 32);

    /* 4 partial products: _mm512_mul_epu32 = 32x32->64, 8 lanes */
    __m512i p_hh = _mm512_mul_epu32(a_hi, b_hi);
    __m512i p_hl = _mm512_mul_epu32(a_hi, b_lo);
    __m512i p_lh = _mm512_mul_epu32(a_lo, b_hi);
    __m512i p_ll = _mm512_mul_epu32(a_lo, b_lo);

    /* carry chain (AVX-512 has mask-based carry detection) */
    __m512i mid = _mm512_add_epi64(p_hl, p_lh);
    __mmask8 mid_carry = _mm512_cmplt_epu64_mask(mid, p_hl);
    __m512i mid_lo = _mm512_and_si512(mid, mask32);
    __m512i mid_hi = _mm512_srli_epi64(mid, 32);
    __m512i t = _mm512_add_epi64(p_ll, _mm512_slli_epi64(mid_lo, 32));
    __mmask8 t_carry = _mm512_cmplt_epu64_mask(t, p_ll);
    __m512i hi = _mm512_add_epi64(p_hh, mid_hi);
    hi = _mm512_add_epi64(hi,
        _mm512_slli_epi64(_mm512_maskz_set1_epi64(mid_carry, 1), 32));
    hi = _mm512_add_epi64(hi,
        _mm512_maskz_set1_epi64(t_carry, 1));
    return hi;
}

__attribute__((noinline))
static void avx512_schoolbook(const uint64_t *in, uint64_t *out,
                              uint64_t op, uint64_t q, uint64_t p, int n) {
    __m512i v_op = _mm512_set1_epi64((long long)op);
    __m512i v_q  = _mm512_set1_epi64((long long)q);
    __m512i v_p  = _mm512_set1_epi64((long long)p);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i x = _mm512_loadu_si512((__m512i*)(in + i));
        __m512i hi = avx512_mulhi_epu64(x, v_q);
        __m512i lo = _mm512_mullo_epi64(x, v_op);
        __m512i r  = _mm512_sub_epi64(lo, _mm512_mullo_epi64(hi, v_p));
        __mmask8 ge = _mm512_cmpge_epu64_mask(r, v_p);
        r = _mm512_mask_sub_epi64(r, ge, r, v_p);
        _mm512_storeu_si512((__m512i*)(out + i), r);
    }
    for (; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (3) AVX-512 schoolbook with 2x interleave
 * ============================================================ */
__attribute__((noinline))
static void avx512_schoolbook_2x(const uint64_t *in, uint64_t *out,
                                  uint64_t op, uint64_t q, uint64_t p, int n) {
    __m512i v_op = _mm512_set1_epi64((long long)op);
    __m512i v_q  = _mm512_set1_epi64((long long)q);
    __m512i v_p  = _mm512_set1_epi64((long long)p);
    int i = 0;
    int full = (n / 16) * 16;
    for (; i < full; i += 16) {
        __m512i x0 = _mm512_loadu_si512((__m512i*)(in + i));
        __m512i x1 = _mm512_loadu_si512((__m512i*)(in + i + 8));
        __m512i h0 = avx512_mulhi_epu64(x0, v_q);
        __m512i h1 = avx512_mulhi_epu64(x1, v_q);
        __m512i lo0 = _mm512_mullo_epi64(x0, v_op);
        __m512i r0 = _mm512_sub_epi64(lo0, _mm512_mullo_epi64(h0, v_p));
        __mmask8 ge0 = _mm512_cmpge_epu64_mask(r0, v_p);
        r0 = _mm512_mask_sub_epi64(r0, ge0, r0, v_p);
        _mm512_storeu_si512((__m512i*)(out + i), r0);
        __m512i lo1 = _mm512_mullo_epi64(x1, v_op);
        __m512i r1 = _mm512_sub_epi64(lo1, _mm512_mullo_epi64(h1, v_p));
        __mmask8 ge1 = _mm512_cmpge_epu64_mask(r1, v_p);
        r1 = _mm512_mask_sub_epi64(r1, ge1, r1, v_p);
        _mm512_storeu_si512((__m512i*)(out + i + 8), r1);
    }
    for (; i + 8 <= n; i += 8) {
        __m512i x = _mm512_loadu_si512((__m512i*)(in + i));
        __m512i hi = avx512_mulhi_epu64(x, v_q);
        __m512i lo = _mm512_mullo_epi64(x, v_op);
        __m512i r  = _mm512_sub_epi64(lo, _mm512_mullo_epi64(hi, v_p));
        __mmask8 ge = _mm512_cmpge_epu64_mask(r, v_p);
        r = _mm512_mask_sub_epi64(r, ge, r, v_p);
        _mm512_storeu_si512((__m512i*)(out + i), r);
    }
    for (; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
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
        {"scalar schoolbook (N=1)",  scalar_schoolbook},
        {"avx512 schoolbook (N=8)",  avx512_schoolbook},
        {"avx512 2x interleave",     avx512_schoolbook_2x},
    };

    printf("=== Pure schoolbook benchmark (x86, no umulh) ===\n");
    printf("    N=%d ITERS=%d VL=512bit(N=8)\n\n", N_ELEMS, ITERS);

    double base = 0;
    for (int t = 0; t < 3; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t0 = rdtsc();
        for (int j = 0; j < ITERS; j++)
            tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t1 = rdtsc();
        double cyc = (double)(t1-t0) / (ITERS * N_ELEMS);
        if (t == 0) base = cyc;
        printf("    %-28s %7.3f cyc/elem  (%.2fx vs scalar)\n",
               tests[t].name, cyc, base/cyc);
    }

    /* correctness */
    uint64_t ref[N_ELEMS], v[N_ELEMS];
    scalar_schoolbook(in, ref, op, q, p, N_ELEMS);
    int ok = 1;
    fn_t others[] = {avx512_schoolbook, avx512_schoolbook_2x};
    const char *names[] = {"avx512", "avx512-2x"};
    for (int t = 0; t < 2; t++) {
        others[t](in, v, op, q, p, N_ELEMS);
        for (int i = 0; i < N_ELEMS; i++) {
            if (v[i] != ref[i]) { printf("    MISMATCH %s[%d]\n", names[t], i); ok=0; break; }
        }
    }
    printf("\n    Correctness: %s\n", ok ? "ALL MATCH" : "FAILED");
    free(in); free(out);
    return 0;
}
