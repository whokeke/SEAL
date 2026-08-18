// Pure schoolbook comparison: scalar vs NEON vs SVE1 vs SVE2
// NO umulh anywhere — all 4 variants use 32-bit decomposition
//
// Build: gcc -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_schoolbook bench_schoolbook.c
// Run:   taskset -c 1 ./bench_schoolbook
//
// Variants:
//   (1) scalar schoolbook — 1 element/iter, no vectorization (force no-tree-vectorize)
//   (2) NEON schoolbook — 2 elements/iter, vmull_u32 (128-bit, N=2)
//   (3) SVE1 schoolbook — 4 elements/iter, mul z.d at 1 inst/cyc (256-bit, N=4)
//   (4) SVE2 schoolbook — 4 elements/iter, svmullb/svmullt at 2 inst/cyc (256-bit, N=4)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   (1000)

static inline uint64_t cntvct(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v;
}

/* ============================================================
 * (1) SCALAR SCHOOLBOOK — prevent auto-vectorization
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
 * (2) NEON SCHOOLBOOK — vmull_u32 (32x32->64), 2 lanes/vector
 * ============================================================ */
__attribute__((noinline))
static void neon_schoolbook(const uint64_t *in, uint64_t *out,
                            uint64_t op, uint64_t q, uint64_t p, int n) {
    int i = 0;
    int full = n & ~1;
    for (; i < full; i += 2) {
        uint64x2_t va = vld1q_u64(in + i);
        uint64x2_t vq = vdupq_n_u64(q);

        uint32x2_t a_lo = vmovn_u64(va);
        uint32x2_t a_hi = vshrn_n_u64(va, 32);
        uint32x2_t q_lo = vmovn_u64(vq);
        uint32x2_t q_hi = vshrn_n_u64(vq, 32);

        uint64x2_t p_ll = vmull_u32(a_lo, q_lo);
        uint64x2_t p_hh = vmull_u32(a_hi, q_hi);
        uint64x2_t p_hl = vmull_u32(a_hi, q_lo);
        uint64x2_t p_lh = vmull_u32(a_lo, q_hi);

        /* Extract to scalar for carry chain (NEON has no carry detection) */
        for (int k = 0; k < 2; k++) {
            uint64_t p_ll_k = vgetq_lane_u64(p_ll, k);
            uint64_t p_hh_k = vgetq_lane_u64(p_hh, k);
            uint64_t p_hl_k = vgetq_lane_u64(p_hl, k);
            uint64_t p_lh_k = vgetq_lane_u64(p_lh, k);
            uint64_t mid = p_hl_k + p_lh_k;
            uint64_t c1 = (mid < p_hl_k);
            uint64_t mid_lo = mid & 0xFFFFFFFF, mid_hi = mid >> 32;
            uint64_t t = p_ll_k + (mid_lo << 32);
            uint64_t c2 = (t < p_ll_k);
            uint64_t hi = p_hh_k + mid_hi + (c1 << 32) + c2;
            uint64_t lo = in[i+k] * op;
            uint64_t r = lo - hi * p;
            out[i+k] = (r >= p) ? r - p : r;
        }
    }
    for (; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (3) SVE1 SCHOOLBOOK — mul z.d (64-bit, only low 32 used), 4 lanes
 * Hand-written: same algorithm as scalar but vectorized with SVE1
 * Uses mul z.d (not umulh!), cmphi for carry, sel for carry value
 * ============================================================ */
static inline svuint64_t sve1_mulhi(svbool_t pg, svuint64_t a, svuint64_t b) {
    svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
    svuint64_t a_lo = svand_u64_x(pg, a, mask32);
    svuint64_t a_hi = svlsr_n_u64_x(pg, a, 32);
    svuint64_t b_lo = svand_u64_x(pg, b, mask32);
    svuint64_t b_hi = svlsr_n_u64_x(pg, b, 32);

    /* 4 partial products: mul z.d (64x64->low64, but operands < 2^32 so result < 2^64) */
    svuint64_t p_ll = svmul_u64_x(pg, a_lo, b_lo);
    svuint64_t p_hh = svmul_u64_x(pg, a_hi, b_hi);
    svuint64_t p_hl = svmul_u64_x(pg, a_hi, b_lo);
    svuint64_t p_lh = svmul_u64_x(pg, a_lo, b_hi);

    /* carry chain (SVE cmphi = unsigned >, cmplt = unsigned <) */
    svuint64_t mid = svadd_u64_x(pg, p_hl, p_lh);
    svbool_t mid_carry = svcmplt_u64(pg, mid, p_hl);  /* mid < p_hl → carry */
    svuint64_t mid_lo = svand_u64_x(pg, mid, mask32);
    svuint64_t mid_hi = svlsr_n_u64_x(pg, mid, 32);
    svuint64_t t = svadd_u64_x(pg, p_ll, svlsl_n_u64_x(pg, mid_lo, 32));
    svbool_t t_carry = svcmplt_u64(pg, t, p_ll);
    svuint64_t one = svdup_n_u64(1);
    svuint64_t zero = svdup_n_u64(0);
    svuint64_t hi = svadd_u64_x(pg, p_hh, mid_hi);
    hi = svadd_u64_x(pg, hi, svlsl_n_u64_x(pg, svsel_u64(mid_carry, one, zero), 32));
    hi = svadd_u64_x(pg, hi, svsel_u64(t_carry, one, zero));
    return hi;
}

__attribute__((noinline))
static void sve1_schoolbook(const uint64_t *in, uint64_t *out,
                             uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t ptrue = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    uint64_t N = svcntd();
    uint64_t full = (n / N) * N;
    uint64_t i = 0;
    for (; i < full; i += N) {
        svuint64_t x = svld1_u64(ptrue, in + i);
        svuint64_t hi = sve1_mulhi(ptrue, x, sv_q);
        svuint64_t lo = svmul_u64_x(ptrue, x, sv_op);
        svuint64_t r  = svmls_u64_x(ptrue, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(ptrue, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(ptrue, out + i, r);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b64((int64_t)i, (int64_t)n);
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve1_mulhi(pg, x, sv_q);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
}

/* ============================================================
 * (4) SVE2 SCHOOLBOOK — svmullb/svmullt (32x32->64 widening), 4 lanes
 * Same as SVE32, but renamed for clarity
 * ============================================================ */
static inline svuint64_t sve2_mulhi(svbool_t pg, svuint64_t a, svuint64_t b,
                                     svuint32_t b32, svuint32_t b_hi32,
                                     svuint64_t mask32, svuint64_t one, svuint64_t zero) {
    svuint32_t a32 = svreinterpret_u32_u64(a);
    svuint64_t p_ll = svmullb_u64(a32, b32);
    svuint64_t p_hh = svmullt_u64(a32, b32);
    svuint64_t a_hi = svlsr_n_u64_x(pg, a, 32);
    svuint32_t a_hi32 = svreinterpret_u32_u64(a_hi);
    svuint64_t p_hl = svmullb_u64(a_hi32, b32);
    svuint64_t p_lh = svmullb_u64(a32, b_hi32);
    svuint64_t mid = svadd_u64_x(pg, p_hl, p_lh);
    svbool_t mc = svcmplt_u64(pg, mid, p_hl);
    svuint64_t mid_lo = svand_u64_x(pg, mid, mask32);
    svuint64_t mid_hi = svlsr_n_u64_x(pg, mid, 32);
    svuint64_t t = svadd_u64_x(pg, p_ll, svlsl_n_u64_x(pg, mid_lo, 32));
    svbool_t tc = svcmplt_u64(pg, t, p_ll);
    svuint64_t hi = svadd_u64_x(pg, p_hh, mid_hi);
    hi = svadd_u64_x(pg, hi, svlsl_n_u64_x(pg, svsel_u64(mc, one, zero), 32));
    hi = svadd_u64_x(pg, hi, svsel_u64(tc, one, zero));
    return hi;
}

__attribute__((noinline))
static void sve2_schoolbook(const uint64_t *in, uint64_t *out,
                             uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t ptrue = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    svuint32_t q32 = svreinterpret_u32_u64(sv_q);
    svuint64_t q_hi = svlsr_n_u64_x(ptrue, sv_q, 32);
    svuint32_t q_hi32 = svreinterpret_u32_u64(q_hi);
    svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
    svuint64_t one = svdup_n_u64(1);
    svuint64_t zero = svdup_n_u64(0);
    uint64_t N = svcntd();
    uint64_t full = (n / N) * N;
    uint64_t i = 0;
    for (; i < full; i += N) {
        svuint64_t x = svld1_u64(ptrue, in + i);
        svuint64_t hi = sve2_mulhi(ptrue, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(ptrue, x, sv_op);
        svuint64_t r  = svmls_u64_x(ptrue, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(ptrue, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(ptrue, out + i, r);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b64((int64_t)i, (int64_t)n);
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve2_mulhi(pg, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
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
        {"neon schoolbook (N=2)",    neon_schoolbook},
        {"sve1 schoolbook (N=4)",    sve1_schoolbook},
        {"sve2 schoolbook (N=4)",    sve2_schoolbook},
    };

    printf("=== Pure schoolbook benchmark (no umulh) ===\n");
    printf("    N=%d ITERS=%d VL=256bit(N=4) HIP12 2.3GHz\n\n", N_ELEMS, ITERS);

    double base = 0;
    for (int t = 0; t < 4; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t0 = cntvct();
        for (int j = 0; j < ITERS; j++)
            tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t1 = cntvct();
        double cyc = (double)(t1-t0) * 23.0 / (ITERS * N_ELEMS);
        if (t == 0) base = cyc;
        printf("    %-28s %7.3f cyc/elem  (%.2fx vs scalar)\n",
               tests[t].name, cyc, base/cyc);
    }

    /* correctness */
    uint64_t ref[N_ELEMS], v[N_ELEMS];
    scalar_schoolbook(in, ref, op, q, p, N_ELEMS);
    int ok = 1;
    fn_t others[] = {neon_schoolbook, sve1_schoolbook, sve2_schoolbook};
    const char *names[] = {"neon", "sve1", "sve2"};
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
