// Benchmark: Lazy Barrett mulmod — AVX-512 vs scalar
// Build: gcc -O3 -mavx512f -mavx512dq -D_GNU_SOURCE -o bench_mulmod_avx512 bench_mulmod_avx512.c
// Run:   taskset -c 1 ./bench_mulmod_avx512
//
// Tests the AVX-512 schoolbook mulhi (avx512_mulhi_epu64) which uses
// _mm512_mul_epu32 (32x32->64 widening) to emulate 64x64->high64.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   1000

static inline uint64_t cntvct(void){
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v;
}

// ---- AVX-512 schoolbook mulhi (same as SEAL's avx512_mulhi_epu64) ----
static inline __m512i avx512_mulhi_epu64(__m512i a, __m512i b) {
    __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFF);
    __m512i a_lo = _mm512_and_si512(a, mask32);
    __m512i a_hi = _mm512_srli_epi64(a, 32);
    __m512i b_lo = _mm512_and_si512(b, mask32);
    __m512i b_hi = _mm512_srli_epi64(b, 32);
    __m512i p_hh = _mm512_mul_epu32(a_hi, b_hi);
    __m512i p_hl = _mm512_mul_epu32(a_hi, b_lo);
    __m512i p_lh = _mm512_mul_epu32(a_lo, b_hi);
    __m512i p_ll = _mm512_mul_epu32(a_lo, b_lo);
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

// ---- AVX-512 mulmod ----
static void avx512_mulmod(const uint64_t *in, uint64_t *out,
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
        // scalar fallback for tail
        uint64_t hi = (uint64_t)((unsigned __int128)in[i] * q >> 64);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

// ---- Scalar (__uint128_t → umulh on AArch64, mul on x86) ----
static void scalar_mulmod(const uint64_t *in, uint64_t *out,
                          uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = (uint64_t)((unsigned __int128)in[i] * q >> 64);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

// ---- Scalar schoolbook (no umulh/__uint128_t) ----
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

static void scalar_schoolbook_mulmod(const uint64_t *in, uint64_t *out,
                                     uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    uint64_t *in = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t *out = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t p = 0xFFFFFFFFFFFFFFC5ULL;
    for (int i = 0; i < N_ELEMS; i++) in[i] = (i * 123456789ULL + 1) % p;
    uint64_t op = 0x123456789ABCDEF0ULL % p;
    uint64_t q = op;

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, int);
    struct { const char *name; fn_t fn; } tests[] = {
        {"scalar schoolbook",      scalar_schoolbook_mulmod},
        {"scalar __uint128_t",      scalar_mulmod},
        {"AVX-512 schoolbook",     avx512_mulmod},
    };

    printf("=== Lazy Barrett mulmod benchmark (AVX-512) ===\n");
    printf("    N=%d ITERS=%d\n\n", N_ELEMS, ITERS);

    double base = 0;
    for (int t = 0; t < 3; t++) {
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

    // correctness
    uint64_t ref[N_ELEMS], v[N_ELEMS];
    scalar_mulmod(in, ref, op, q, p, N_ELEMS);
    avx512_mulmod(in, v, op, q, p, N_ELEMS);
    int ok = 1;
    for (int i = 0; i < N_ELEMS; i++) if (v[i] != ref[i]) { ok = 0; break; }
    printf("\n    Correctness: %s\n", ok ? "ALL MATCH" : "FAILED");
    free(in); free(out);
    return 0;
}
