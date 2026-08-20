#include <stdio.h>
#include <stdint.h>
#include <sched.h>

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
#define ITERS 2000000
#define B8(z0,z1,z2,z3,z4,z5,z6,z7) \
    __asm__ volatile("" : "+r"(z0),"+r"(z1),"+r"(z2),"+r"(z3),"+r"(z4),"+r"(z5),"+r"(z6),"+r"(z7))

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;
    uint64_t c = 0x0F1E2D3C4B5A6978ULL;
    uint64_t d = 0x13579BDF2468ACE0ULL;
    uint64_t t0, t1, r;
    /* 8 independent registers for throughput test */
    uint64_t x0=a,x1=a,x2=a,x3=a,x4=a,x5=a,x6=a,x7=a;

    printf("=== mul+umulh fusion: gap insertion (THROUGHPUT) ===\n");
    printf("    8 pairs/iter, independent registers\n");
    printf("    If macro-op fusion: gap breaks it → ~2.0 cyc/inst\n");
    printf("    If data forwarding: gap doesn't break → ~1.0 cyc/inst\n\n");

    // 1. Baseline: mul+umulh adjacent, same operands (throughput)
    x0=a;x1=a;x2=a;x3=a;x4=a;x5=a;x6=a;x7=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n" "umulh %1, %2, %3\n"
            "mul %4, %6, %7\n" "umulh %5, %6, %7\n"
            : "=&r"(x0),"=&r"(x1),"=&r"(x2),"=&r"(x3)
            : "0"(x0),"1"(x1),"r"(x4),"r"(b),"r"(x5),"r"(b));
        /* Do 4 more pairs manually to get 8 total */
        __asm__ volatile(
            "mul %0, %2, %3\n" "umulh %1, %2, %3\n"
            "mul %4, %6, %7\n" "umulh %5, %6, %7\n"
            : "=&r"(x2),"=&r"(x3),"=&r"(x4),"=&r"(x5)
            : "0"(x2),"1"(x3),"r"(x6),"r"(b),"r"(x7),"r"(b));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  mul+umulh adjacent       %5.3f cyc/pair\n",
        (double)(t1-t0)*23.0/(ITERS*8));

    // 2. mul+NOP+umulh (1 gap)
    x0=a;x1=a;x2=a;x3=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n" "nop\n" "umulh %1, %2, %3\n"
            "mul %0, %2, %3\n" "nop\n" "umulh %1, %2, %3\n"
            : "=&r"(x0),"=&r"(x1) : "r"(x0),"r"(b),"r"(x2),"r"(x3),"r"(b));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  mul+NOP+umulh (1 gap)    %5.3f cyc/pair  ← break = macro-fusion\n",
        (double)(t1-t0)*23.0/(ITERS*4));

    // 3. mul+2NOP+umulh (2 gap)
    x0=a;x1=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n" "nop\n" "nop\n" "umulh %1, %2, %3\n"
            : "=&r"(x0),"=&r"(x1) : "r"(x0),"r"(b),"r"(x2),"r"(b));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  mul+2NOP+umulh (2 gap)   %5.3f cyc/pair\n",
        (double)(t1-t0)*23.0/(ITERS*2));

    // 4. mul+ADD+umulh (ALU gap, different unit)
    x0=a;x1=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "add %4, %5, %6\n"
            "umulh %1, %2, %3\n"
            : "=&r"(x0),"=&r"(x1),"=&r"(r)
            : "r"(x0),"r"(b),"r"(c),"r"(d));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  mul+ADD+umulh (ALU gap)  %5.3f cyc/pair  ← ALU, diff unit\n",
        (double)(t1-t0)*23.0/(ITERS*2));

    // 5. mul+MUL+umulh (multiply gap, same unit!)
    x0=a;x1=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n"
            "mul %4, %5, %6\n"
            "umulh %1, %2, %3\n"
            : "=&r"(x0),"=&r"(x1),"=&r"(r)
            : "r"(x0),"r"(b),"r"(c),"r"(d));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  mul+MUL+umulh (mul gap)  %5.3f cyc/pair  ← same mul unit\n",
        (double)(t1-t0)*23.0/(ITERS*2));

    // 6. Reference: mul alone (throughput)
    x0=a;x1=a;x2=a;x3=a;x4=a;x5=a;x6=a;x7=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "mul %0, %2, %3\n" "mul %1, %4, %5\n"
            "mul %6, %2, %3\n" "mul %7, %4, %5\n"
            : "=&r"(x0),"=&r"(x1),"=&r"(x2),"=&r"(x3)
            : "0"(x0),"1"(x1),"r"(x4),"r"(b),"r"(x5),"r"(b),"r"(x6),"r"(x7));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("\n  mul alone (reference)    %5.3f cyc/inst\n",
        (double)(t1-t0)*23.0/(ITERS*4));

    // 7. umulh alone (throughput)
    x0=a;x1=a;x2=a;x3=a;
    t0=cntvct();
    for (int i=0;i<ITERS;i++){
        __asm__ volatile(
            "umulh %0, %2, %3\n" "umulh %1, %4, %5\n"
            : "=&r"(x0),"=&r"(x1)
            : "r"(x0),"r"(b),"r"(x2),"r"(b));
        B8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    t1=cntvct();
    printf("  umulh alone (reference)   %5.3f cyc/inst\n",
        (double)(t1-t0)*23.0/(ITERS*2));

    printf("\n=== Analysis ===\n");
    printf("  adjacent = NOP-gap = ALU-gap → data forwarding (not decoder fusion)\n");
    printf("  adjacent < NOP-gap → macro-op fusion (decoder fuses adjacent pairs)\n");
    printf("  mul-gap > ALU-gap → multiply unit resource conflict\n");

    return 0;
}
