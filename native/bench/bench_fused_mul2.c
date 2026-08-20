#include <stdio.h>
#include <stdint.h>
#include <sched.h>

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
#define ITERS 10000000
#define BARRIER(z) __asm__ volatile("" : "+r"(z))

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;
    uint64_t c = 0x0F1E2D3C4B5A6978ULL;  // different from b
    uint64_t p = 0xFFFFFFFFFFFFFFC5ULL;
    uint64_t t0, t1, lo, hi, r;

    printf("=== mul/umulh fusion: same vs different operands ===\n\n");

    // 1. mul+umulh, SAME both operands: mul(x,b) + umulh(x,b)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "umulh %1, %2, %3\n"
            : "=&r"(lo), "=r"(hi) : "r"(a), "r"(b)
        );
        r = lo ^ hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul(x,b)+umulh(x,b)  same op  %5.2f cyc  ← fusion?\n", (double)(t1-t0)*23.0/ITERS);

    // 2. mul+umulh, DIFFERENT second operand: mul(x,b) + umulh(x,c)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "umulh %1, %2, %4\n"
            : "=&r"(lo), "=r"(hi) : "r"(a), "r"(b), "r"(c)
        );
        r = lo ^ hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul(x,b)+umulh(x,c)  diff op  %5.2f cyc  ← no fusion\n", (double)(t1-t0)*23.0/ITERS);

    // 3. mul+umulh, COMPLETELY different: mul(x,b) + umulh(c,d)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "umulh %1, %4, %5\n"
            : "=&r"(lo), "=r"(hi) : "r"(a), "r"(b), "r"(c), "r"(p)
        );
        r = lo ^ hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul(x,b)+umulh(c,d)  all diff %5.2f cyc  ← independent\n", (double)(t1-t0)*23.0/ITERS);

    // 4. Just mul (reference latency)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile("mul %0, %1, %2" : "=r"(lo) : "r"(a), "r"(b));
        r = lo; BARRIER(r);
    }
    t1 = cntvct();
    printf("  mul alone            (ref)    %5.2f cyc\n", (double)(t1-t0)*23.0/ITERS);

    // 5. Just umulh (reference latency)
    r = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile("umulh %0, %1, %2" : "=r"(hi) : "r"(a), "r"(b));
        r = hi; BARRIER(r);
    }
    t1 = cntvct();
    printf("  umulh alone          (ref)    %5.2f cyc\n", (double)(t1-t0)*23.0/ITERS);

    printf("\n=== Full Barrett: same vs different op/q ===\n\n");

    // 6. Barrett with op=q (same → fusion possible)
    uint64_t x = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %3, %4\n"       // lo = x * op
            "umulh %1, %3, %4\n"      // hi = umulh(x, op)  ← same operands!
            "msub %0, %1, %5, %0\n"  // lo = lo - hi * p
            : "=&r"(lo), "=&r"(hi)
            : "0"(lo), "r"(x), "r"(b), "r"(p)  // op = q = b
        );
        x = lo; BARRIER(x);
    }
    t1 = cntvct();
    printf("  Barrett op=q (fused)  %5.2f cyc  ← mul+umulh same op\n", (double)(t1-t0)*23.0/ITERS);

    // 7. Barrett with op≠q (different → no fusion)
    x = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mul %0, %3, %4\n"       // lo = x * op
            "umulh %1, %3, %6\n"     // hi = umulh(x, q)  ← different operands!
            "msub %0, %1, %5, %0\n"  // lo = lo - hi * p
            : "=&r"(lo), "=&r"(hi)
            : "0"(lo), "r"(x), "r"(b), "r"(p), "r"(c)  // op=b, q=c, b≠c
        );
        x = lo; BARRIER(x);
    }
    t1 = cntvct();
    printf("  Barrett op≠q (no fuse) %5.2f cyc  ← mul+umulh diff op\n", (double)(t1-t0)*23.0/ITERS);

    // 8. Barrett reversed order: umulh first, then mul
    x = a; t0 = cntvct();
    for (int i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "umulh %1, %3, %6\n"     // hi = umulh(x, q)  first
            "mul %0, %3, %4\n"       // lo = x * op       second
            "msub %0, %1, %5, %0\n"  // lo = lo - hi * p
            : "=&r"(lo), "=&r"(hi)
            : "0"(lo), "r"(x), "r"(b), "r"(p), "r"(c)
        );
        x = lo; BARRIER(x);
    }
    t1 = cntvct();
    printf("  Barrett reversed order %5.2f cyc  ← umulh first\n", (double)(t1-t0)*23.0/ITERS);

    return 0;
}
