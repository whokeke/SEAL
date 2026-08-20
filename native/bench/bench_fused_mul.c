#include <stdio.h>
#include <stdint.h>
#include <sched.h>

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
#define ITERS 10000000
#define BARRIER(z) __asm__ volatile("" : "+r"(z))

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    volatile uint64_t a = 0x123456789ABCDEF0ULL;
    volatile uint64_t b = 0xFEDCBA9876543210ULL;
    uint64_t r;
    uint64_t t0, t1;

    printf("=== Scalar mul/umulh fusion test (HIP12) ===\n\n");

    // 1. mul alone
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) { r = r * b; BARRIER(r); }
    t1 = cntvct();
    printf("  mul alone           %7.2f cyc\n", (double)(t1-t0)*23.0/ITERS);

    // 2. umulh alone
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile("umulh %0, %1, %2" : "=r"(r) : "r"(r), "r"(b));
        BARRIER(r);
    }
    t1 = cntvct();
    printf("  umulh alone         %7.2f cyc\n", (double)(t1-t0)*23.0/ITERS);

    // 3. mul + umulh, SAME operands
    uint64_t lo, hi;
    lo = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "umulh %1, %2, %3\n"
            : "=&r"(lo), "=r"(hi) : "r"(lo), "r"(b)
        );
        r = lo ^ hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul+umulh same op   %7.2f cyc  (fused → ~4.0)\n", (double)(t1-t0)*23.0/ITERS);

    // 4. mul + umulh, DIFFERENT operands
    t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "umulh %1, %2, %3\n"
            : "=&r"(lo), "=r"(hi) : "r"(a), "r"(b)
        );
        r = lo ^ hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul+umulh diff op   %7.2f cyc  (same operands, const input)\n", (double)(t1-t0)*23.0/ITERS);

    // 5. madd (fused mul-add)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile("madd %0, %1, %2, %0" : "=r"(r) : "r"(r), "r"(b), "0"(r));
        BARRIER(r);
    }
    t1 = cntvct();
    printf("  madd (mul+add fused)%7.2f cyc  (reference)\n", (double)(t1-t0)*23.0/ITERS);

    // 6. mul + add (unfused)
    r = a; uint64_t s = 0; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        uint64_t m = r * b;
        s = m + s;
        r = s; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul+add (unfused)   %7.2f cyc  (reference)\n", (double)(t1-t0)*23.0/ITERS);

    // 7. msub (fused mul-sub)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile("msub %0, %1, %2, %0" : "=r"(r) : "r"(r), "r"(b), "0"(r));
        BARRIER(r);
    }
    t1 = cntvct();
    printf("  msub (mul-sub fused)%7.2f cyc  (reference)\n", (double)(t1-t0)*23.0/ITERS);

    // 8. Full Barrett: mul + umulh + msub
    uint64_t x = a, op = b, q = b, p = 0xFFFFFFFFFFFFFFC5ULL;
    t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        uint64_t lo2, hi2;
        __asm__ volatile(
            "mul %0, %3, %4\n"
            "umulh %1, %3, %5\n"
            "msub %0, %1, %6, %0\n"
            : "=&r"(lo2), "=&r"(hi2)
            : "0"(lo2), "r"(x), "r"(op), "r"(q), "r"(p)
        );
        x = lo2; BARRIER(x);
    }
    t1 = cntvct();
    printf("\n  Barrett mul+umulh+msub  %7.2f cyc\n", (double)(t1-t0)*23.0/ITERS);
    printf("  mul+umulh fused →       ~8 cyc (2 mul)\n");
    printf("  not fused →             ~12 cyc (3 mul)\n");

    return 0;
}
