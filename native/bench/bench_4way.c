// Comprehensive mulmod benchmark: 4 variants × 2 compilers
// (1) scalar-schoolbook  (2) scalar-int128  (3) vector-schoolbook  (4) vector-int128(svmulh)
//
// Build with GCC 15:
//   /home/hukeke/spec/toolchains/gcc-15/bin/aarch64-unknown-linux-gnu-gcc -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_4way_gcc15 bench_4way.c
// Build with LLVM 22:
//   /home/hukeke/spec/toolchains/LLVM-22.1.8-release/bin/clang -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_4way_llvm22 bench_4way.c
//
// Also generate assembly:
//   ... -S -o bench_4way_gcc15.s
//   ... -S -o bench_4way_llvm22.s

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   (1000)

static inline uint64_t cntvct(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v;
}

/* ============================================================
 * (1) SCALAR SCHOOLBOOK — 32-bit decomposition, ~16 instructions
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
                              uint64_t op, uint64_t q, uint64_t p, uint64_t n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = mulhi_schoolbook(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (2) SCALAR __uint128_t — compiler generates umulh+mul+msub
 * ============================================================ */
__attribute__((noinline))
static void scalar_int128(const uint64_t *in, uint64_t *out,
                          uint64_t op, uint64_t q, uint64_t p, uint64_t n) {
    for (int i = 0; i < n; i++) {
        unsigned __int128 prod = (unsigned __int128)in[i] * q;
        uint64_t hi = (uint64_t)(prod >> 64);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ============================================================
 * (3) VECTOR SCHOOLBOOK — SVE32 svmullb/svmullt + carry chain
 * ============================================================ */
static inline svuint64_t sve32_mulhi(svbool_t pg, svuint64_t a, svuint64_t b,
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
static void vector_schoolbook(const uint64_t *in, uint64_t *out,
                              uint64_t op, uint64_t q, uint64_t p, uint64_t n) {
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
        svuint64_t hi = sve32_mulhi(ptrue, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(ptrue, x, sv_op);
        svuint64_t r  = svmls_u64_x(ptrue, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(ptrue, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(ptrue, out + i, r);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve32_mulhi(pg, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
}

/* ============================================================
 * (4) VECTOR INT128 — SVE native svmulh_u64_x (1 instruction mulhi)
 * ============================================================ */
__attribute__((noinline))
static void vector_int128(const uint64_t *in, uint64_t *out,
                          uint64_t op, uint64_t q, uint64_t p, uint64_t n) {
    svbool_t ptrue = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    uint64_t N = svcntd();
    uint64_t full = (n / N) * N;
    uint64_t i = 0;
    for (; i < full; i += N) {
        svuint64_t x = svld1_u64(ptrue, in + i);
        svuint64_t hi = svmulh_u64_x(ptrue, x, sv_q);
        svuint64_t lo = svmul_u64_x(ptrue, x, sv_op);
        svuint64_t r  = svmls_u64_x(ptrue, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(ptrue, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(ptrue, out + i, r);
    }
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = svmulh_u64_x(pg, x, sv_q);
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

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t);
    struct { const char *name; fn_t fn; } tests[] = {
        {"scalar schoolbook",   scalar_schoolbook},
        {"scalar __uint128_t",   scalar_int128},
        {"vector schoolbook",    vector_schoolbook},
        {"vector int128(svmulh)",vector_int128},
    };

    printf("=== 4-way mulmod benchmark ===\n");
    printf("    N=%d ITERS=%d VL=%dbit(N=%d) HIP12 2.3GHz\n\n",
           N_ELEMS, ITERS, (int)svcntb()*8, (int)svcntd());

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
    fn_t others[] = {scalar_schoolbook, vector_schoolbook, vector_int128};
    const char *names[] = {"schoolbook", "vec-schoolbook", "vec-int128"};
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
