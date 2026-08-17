#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   1000

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }

/* ---- SVE native (svmulh) ---- */
static void sve_mulmod(const uint64_t *in, uint64_t *out,
                       uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t pg = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    int N = svcntd();
    int i = 0;
    for (; i + N <= n; i += N) {
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = svmulh_u64_x(pg, x, sv_q);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
    if (i < n) {
        svbool_t pg2 = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg2, in + i);
        svuint64_t hi = svmulh_u64_x(pg2, x, sv_q);
        svuint64_t lo = svmul_u64_x(pg2, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg2, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg2, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg2, out + i, r);
    }
}

/* ---- SVE32 schoolbook mulhi ---- */
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

/* ---- SVE32 no interleave ---- */
static void sve32_mulmod(const uint64_t *in, uint64_t *out,
                         uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t pg = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    svuint32_t q32 = svreinterpret_u32_u64(sv_q);
    svuint64_t q_hi = svlsr_n_u64_x(pg, sv_q, 32);
    svuint32_t q_hi32 = svreinterpret_u32_u64(q_hi);
    svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
    svuint64_t one = svdup_n_u64(1);
    svuint64_t zero = svdup_n_u64(0);
    int N = svcntd();
    int i = 0;
    for (; i + N <= n; i += N) {
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve32_mulhi(pg, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
    if (i < n) {
        svbool_t pg2 = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg2, in + i);
        svuint64_t hi = sve32_mulhi(pg2, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg2, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg2, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg2, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg2, out + i, r);
    }
}

/* ---- SVE32 2x interleave ---- */
static void sve32_2x_mulmod(const uint64_t *in, uint64_t *out,
                            uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t pg = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    svuint32_t q32 = svreinterpret_u32_u64(sv_q);
    svuint64_t q_hi = svlsr_n_u64_x(pg, sv_q, 32);
    svuint32_t q_hi32 = svreinterpret_u32_u64(q_hi);
    svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
    svuint64_t one = svdup_n_u64(1);
    svuint64_t zero = svdup_n_u64(0);
    int N = svcntd();
    int full = (n / (2*N)) * (2*N);
    int i = 0;
    for (; i < full; i += 2*N) {
        /* Load 2 independent vectors */
        svuint64_t x0 = svld1_u64(pg, in + i);
        svuint64_t x1 = svld1_u64(pg, in + i + N);
        /* mulhi for both — carry chains are independent */
        svuint64_t h0 = sve32_mulhi(pg, x0, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t h1 = sve32_mulhi(pg, x1, sv_q, q32, q_hi32, mask32, one, zero);
        /* finish mulmod */
        svuint64_t lo0 = svmul_u64_x(pg, x0, sv_op);
        svuint64_t r0 = svmls_u64_x(pg, lo0, sv_p, h0);
        svbool_t ge0 = svcmpge_u64(pg, r0, sv_p);
        r0 = svsub_u64_m(ge0, r0, sv_p);
        svst1_u64(pg, out + i, r0);

        svuint64_t lo1 = svmul_u64_x(pg, x1, sv_op);
        svuint64_t r1 = svmls_u64_x(pg, lo1, sv_p, h1);
        svbool_t ge1 = svcmpge_u64(pg, r1, sv_p);
        r1 = svsub_u64_m(ge1, r1, sv_p);
        svst1_u64(pg, out + i + N, r1);
    }
    /* tail */
    for (; i + N <= n; i += N) {
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve32_mulhi(pg, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
    if (i < n) {
        svbool_t pg2 = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg2, in + i);
        svuint64_t hi = sve32_mulhi(pg2, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg2, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg2, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg2, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg2, out + i, r);
    }
}

/* ---- SVE32 4x interleave ---- */
static void sve32_4x_mulmod(const uint64_t *in, uint64_t *out,
                            uint64_t op, uint64_t q, uint64_t p, int n) {
    svbool_t pg = svptrue_b64();
    svuint64_t sv_op = svdup_n_u64(op);
    svuint64_t sv_q  = svdup_n_u64(q);
    svuint64_t sv_p  = svdup_n_u64(p);
    svuint32_t q32 = svreinterpret_u32_u64(sv_q);
    svuint64_t q_hi = svlsr_n_u64_x(pg, sv_q, 32);
    svuint32_t q_hi32 = svreinterpret_u32_u64(q_hi);
    svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
    svuint64_t one = svdup_n_u64(1);
    svuint64_t zero = svdup_n_u64(0);
    int N = svcntd();
    int full = (n / (4*N)) * (4*N);
    int i = 0;
    for (; i < full; i += 4*N) {
        svuint64_t x0 = svld1_u64(pg, in + i);
        svuint64_t x1 = svld1_u64(pg, in + i + N);
        svuint64_t x2 = svld1_u64(pg, in + i + 2*N);
        svuint64_t x3 = svld1_u64(pg, in + i + 3*N);
        svuint64_t h0 = sve32_mulhi(pg, x0, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t h1 = sve32_mulhi(pg, x1, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t h2 = sve32_mulhi(pg, x2, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t h3 = sve32_mulhi(pg, x3, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo0 = svmul_u64_x(pg, x0, sv_op);
        svuint64_t r0 = svmls_u64_x(pg, lo0, sv_p, h0);
        svbool_t ge0 = svcmpge_u64(pg, r0, sv_p);
        r0 = svsub_u64_m(ge0, r0, sv_p);
        svst1_u64(pg, out + i, r0);
        svuint64_t lo1 = svmul_u64_x(pg, x1, sv_op);
        svuint64_t r1 = svmls_u64_x(pg, lo1, sv_p, h1);
        svbool_t ge1 = svcmpge_u64(pg, r1, sv_p);
        r1 = svsub_u64_m(ge1, r1, sv_p);
        svst1_u64(pg, out + i + N, r1);
        svuint64_t lo2 = svmul_u64_x(pg, x2, sv_op);
        svuint64_t r2 = svmls_u64_x(pg, lo2, sv_p, h2);
        svbool_t ge2 = svcmpge_u64(pg, r2, sv_p);
        r2 = svsub_u64_m(ge2, r2, sv_p);
        svst1_u64(pg, out + i + 2*N, r2);
        svuint64_t lo3 = svmul_u64_x(pg, x3, sv_op);
        svuint64_t r3 = svmls_u64_x(pg, lo3, sv_p, h3);
        svbool_t ge3 = svcmpge_u64(pg, r3, sv_p);
        r3 = svsub_u64_m(ge3, r3, sv_p);
        svst1_u64(pg, out + i + 3*N, r3);
    }
    for (; i + N <= n; i += N) {
        svuint64_t x = svld1_u64(pg, in + i);
        svuint64_t hi = sve32_mulhi(pg, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg, out + i, r);
    }
    if (i < n) {
        svbool_t pg2 = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg2, in + i);
        svuint64_t hi = sve32_mulhi(pg2, x, sv_q, q32, q_hi32, mask32, one, zero);
        svuint64_t lo = svmul_u64_x(pg2, x, sv_op);
        svuint64_t r  = svmls_u64_x(pg2, lo, sv_p, hi);
        svbool_t ge = svcmpge_u64(pg2, r, sv_p);
        r = svsub_u64_m(ge, r, sv_p);
        svst1_u64(pg2, out + i, r);
    }
}

/* ---- Scalar ---- */
static inline uint64_t umulh(uint64_t a, uint64_t b) {
    uint64_t r; __asm__ volatile("umulh %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); return r;
}

static void scalar_mulmod(const uint64_t *in, uint64_t *out,
                          uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = umulh(in[i], q);
        uint64_t lo = in[i] * op;
        uint64_t r = lo - hi * p;
        out[i] = (r >= p) ? r - p : r;
    }
}

/* ---- Benchmark ---- */
int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    uint64_t *in  = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t *out = aligned_alloc(64, N_ELEMS * sizeof(uint64_t));
    uint64_t p = 0xFFFFFFFFFFFFFFC5ULL;
    for (int i = 0; i < N_ELEMS; i++) in[i] = (i * 123456789ULL + 1) % p;
    uint64_t op = 0x123456789ABCDEF0ULL % p;
    uint64_t q  = op; /* simplified quotient for benchmark */

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, int);
    struct { const char *name; fn_t fn; } tests[] = {
        {"scalar (__umulh)",       scalar_mulmod},
        {"SVE native (svmulh)",    sve_mulmod},
        {"SVE32 (no interleave)",  sve32_mulmod},
        {"SVE32 (2x interleave)",  sve32_2x_mulmod},
        {"SVE32 (4x interleave)",  sve32_4x_mulmod},
    };

    printf("=== Lazy Barrett mulmod benchmark ===\n");
    printf("    N_ELEMS=%d ITERS=%d VL=256bit(N=4) CPU=2.3GHz HIP12\n\n", N_ELEMS, ITERS);

    double base = 0;
    for (int t = 0; t < 5; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS); /* warmup */
        uint64_t t0 = cntvct();
        for (int j = 0; j < ITERS; j++)
            tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t1 = cntvct();
        double cyc = (double)(t1-t0) * 23.0 / (ITERS * N_ELEMS);
        if (t == 0) base = cyc;
        printf("    %-25s %7.3f cyc/elem  (%.2fx vs scalar)\n", tests[t].name, cyc, base/cyc);
    }

    /* correctness */
    uint64_t ref[N_ELEMS], v[N_ELEMS];
    scalar_mulmod(in, ref, op, q, p, N_ELEMS);
    const char *names[] = {"", "", "SVE", "SVE32", "SVE32-2x"};
    fn_t fns[] = {NULL, NULL, sve_mulmod, sve32_mulmod, sve32_2x_mulmod};
    int ok = 1;
    for (int t = 2; t < 5; t++) {
        fns[t](in, v, op, q, p, N_ELEMS);
        for (int i = 0; i < N_ELEMS; i++) {
            if (v[i] != ref[i]) { printf("    MISMATCH %s[%d]\n", names[t], i); ok=0; break; }
        }
    }
    printf("\n    Correctness: %s\n", ok ? "ALL MATCH" : "FAILED");
    free(in); free(out);
    return 0;
}
