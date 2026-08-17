#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>

#define N_ELEMS 65536
#define ITERS 1000

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }

/* 修复前: schoolbook 手动 32-bit 分解 */
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

/* 修复后: __uint128_t → umulh */
static inline uint64_t mulhi_u128(uint64_t a, uint64_t b) {
    return (uint64_t)((unsigned __int128)a * b >> 64);
}

/* 标量 lazy Barrett mulmod */
#define MULMOD_FUNC(name, mulhi_fn) \
static void name(const uint64_t *in, uint64_t *out, uint64_t op, uint64_t q, uint64_t p, int n) { \
    for (int i = 0; i < n; i++) { \
        uint64_t hi = mulhi_fn(in[i], q); \
        uint64_t lo = in[i] * op; \
        uint64_t r = lo - hi * p; \
        out[i] = (r >= p) ? r - p : r; \
    } \
}

MULMOD_FUNC(mulmod_schoolbook, mulhi_schoolbook)
MULMOD_FUNC(mulmod_u128, mulhi_u128)

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    static uint64_t in[N_ELEMS], out[N_ELEMS];
    uint64_t p = 0xFFFFFFFFFFFFFFC5ULL;
    for (int i = 0; i < N_ELEMS; i++) in[i] = (i * 123456789ULL + 1) % p;
    uint64_t op = 0x123456789ABCDEF0ULL % p;
    uint64_t q = op;

    typedef void (*fn_t)(const uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, int);
    struct { const char *name; fn_t fn; } tests[] = {
        {"schoolbook (修复前)", mulmod_schoolbook},
        {"__uint128_t (修复后)", mulmod_u128},
    };

    printf("=== 标量 mulmod: schoolbook vs __uint128_t ===\n");
    printf("    N=%d ITERS=%d CPU=2.3GHz GCC-12 -O3\n\n", N_ELEMS, ITERS);

    for (int t = 0; t < 2; t++) {
        tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t0 = cntvct();
        for (int j = 0; j < ITERS; j++)
            tests[t].fn(in, out, op, q, p, N_ELEMS);
        uint64_t t1 = cntvct();
        double cyc = (double)(t1-t0) * 23.0 / (ITERS * N_ELEMS);
        printf("    %-25s %7.3f cyc/elem\n", tests[t].name, cyc);
    }

    // correctness
    uint64_t r0[N_ELEMS], r1[N_ELEMS];
    mulmod_schoolbook(in, r0, op, q, p, N_ELEMS);
    mulmod_u128(in, r1, op, q, p, N_ELEMS);
    int ok = 1;
    for (int i = 0; i < N_ELEMS; i++) if (r0[i] != r1[i]) { ok = 0; break; }
    printf("\n    Correctness: %s\n", ok ? "MATCH" : "FAILED");
    return 0;
}
