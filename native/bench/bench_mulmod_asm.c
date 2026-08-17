// Benchmark: Lazy Barrett mulmod — SVE inline assembly + scalar
// Build: gcc -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_mulmod_asm bench_mulmod_asm.c
// Run:   taskset -c 1 ./bench_mulmod_asm
//
// Uses raw SVE assembly (umulh/mul/mls/cmphs/sub) via inline asm,
// bypassing the compiler's intrinsic handling entirely.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <sched.h>

#define N_ELEMS (65536)
#define ITERS   1000

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }

/* ---- SVE native inline asm ---- */
/* umulh + mul + mls + cmphs + sub, all in one asm block per vector */
static void sve_asm_mulmod(const uint64_t *in, uint64_t *out,
                            uint64_t op, uint64_t q, uint64_t p, int n) {
    int N = svcntd();
    int i = 0;
    for (; i + N <= n; i += N) {
        __asm__ volatile(
            "ptrue p0.d\n"
            "ld1d  z0.d, p0/z, [%[in]]\n"
            "dup   z1.d, %[q]\n"
            "dup   z2.d, %[op]\n"
            "dup   z3.d, %[p]\n"
            /* hi = umulh(x, q) */
            "mov   z4.d, z0.d\n"
            "umulh z4.d, p0/m, z4.d, z1.d\n"
            /* lo = mul(x, op) */
            "mov   z5.d, z0.d\n"
            "mul   z5.d, p0/m, z5.d, z2.d\n"
            /* r = lo - p*hi */
            "mls   z5.d, p0/m, z3.d, z4.d\n"
            /* if (r >= p) r -= p  (cmphs = unsigned >=) */
            "cmphs p1.d, p0/z, z5.d, z3.d\n"
            "sub   z5.d, p1/m, z5.d, z3.d\n"
            "st1d  z5.d, p0, [%[out]]\n"
            :
            : [in] "r"(in), [out] "r"(out), [q] "r"(q), [op] "r"(op), [p] "r"(p)
            : "z0","z1","z2","z3","z4","z5","p0","p1","memory"
        );
        in += N; out += N;
    }
    /* tail (uses intrinsics for predicate handling) */
    if (i < n) {
        svbool_t pg = svwhilelt_b64(i, n);
        svuint64_t x = svld1_u64(pg, in);
        svuint64_t hi = svmulh_u64_x(pg, x, svdup_n_u64(q));
        svuint64_t lo = svmul_u64_x(pg, x, svdup_n_u64(op));
        svuint64_t r  = svmls_u64_x(pg, lo, svdup_n_u64(p), hi);
        svbool_t ge = svcmpge_u64(pg, r, svdup_n_u64(p));
        r = svsub_u64_m(ge, r, svdup_n_u64(p));
        svst1_u64(pg, out, r);
    }
}



/* ---- Scalar (inline asm umulh) ---- */
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

/* ---- Scalar (__uint128_t) ---- */
static void scalar_u128_mulmod(const uint64_t *in, uint64_t *out,
                               uint64_t op, uint64_t q, uint64_t p, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t hi = (uint64_t)((unsigned __int128)in[i] * q >> 64);
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
    uint64_t q  = op;

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, int);
    struct { const char *name; fn_t fn; } tests[] = {
        {"scalar (inline asm)",       scalar_mulmod},
        {"scalar (__uint128_t)",      scalar_u128_mulmod},
        {"SVE native (inline asm)",   sve_asm_mulmod},
    };

    printf("=== Lazy Barrett mulmod benchmark (SVE inline asm) ===\n");
    printf("    N_ELEMS=%d ITERS=%d VL=%dbit(N=%d) CPU=2.3GHz HIP12\n\n",
           N_ELEMS, ITERS, svcntb()*8, (int)svcntd());

    double base = 0;
    for (int t = 0; t < 3; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS);
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
    const char *names[] = {"", "", "SVE-asm"};
    fn_t fns[] = {NULL, NULL, sve_asm_mulmod};
    int ok = 1;
    for (int t = 2; t < 3; t++) {
        fns[t](in, v, op, q, p, N_ELEMS);
        for (int i = 0; i < N_ELEMS; i++) {
            if (v[i] != ref[i]) { printf("    MISMATCH %s[%d]\n", names[t], i); ok=0; break; }
        }
    }
    printf("\n    Correctness: %s\n", ok ? "ALL MATCH" : "FAILED");
    free(in); free(out);
    return 0;
}
