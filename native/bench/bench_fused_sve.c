#include <stdio.h>
#include <stdint.h>
#include <arm_sve.h>
#include <sched.h>

static inline uint64_t cntvct(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
#define ITERS 2000000
#define BARRIER8(z0,z1,z2,z3,z4,z5,z6,z7) \
    __asm__ volatile("" : "+w"(z0),"+w"(z1),"+w"(z2),"+w"(z3),"+w"(z4),"+w"(z5),"+w"(z6),"+w"(z7))

int main() {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    svbool_t pg = svptrue_b64();
    svuint64_t a = svdup_n_u64(123456789ULL);
    svuint64_t b = svdup_n_u64(987654321ULL);
    svuint64_t c = svdup_n_u64(555555555ULL);  // different from b
    uint64_t t0, t1;
    double per;

    printf("=== SVE mul+umulh fusion test (HIP12, VL=256, N=4) ===\n\n");

    // 1. svmul alone (8 independent)
    {
        svuint64_t z0=a,z1=a,z2=a,z3=a,z4=a,z5=a,z6=a,z7=a;
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            z0=svmul_u64_x(pg,z0,b);z1=svmul_u64_x(pg,z1,b);z2=svmul_u64_x(pg,z2,b);z3=svmul_u64_x(pg,z3,b);
            z4=svmul_u64_x(pg,z4,b);z5=svmul_u64_x(pg,z5,b);z6=svmul_u64_x(pg,z6,b);z7=svmul_u64_x(pg,z7,b);
            BARRIER8(z0,z1,z2,z3,z4,z5,z6,z7);
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/(ITERS*8);
        printf("  svmul x8 (alone)         %5.3f cyc/inst\n", per);
    }

    // 2. svmulh alone (8 independent)
    {
        svuint64_t z0=a,z1=a,z2=a,z3=a,z4=a,z5=a,z6=a,z7=a;
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            z0=svmulh_u64_x(pg,z0,b);z1=svmulh_u64_x(pg,z1,b);z2=svmulh_u64_x(pg,z2,b);z3=svmulh_u64_x(pg,z3,b);
            z4=svmulh_u64_x(pg,z4,b);z5=svmulh_u64_x(pg,z5,b);z6=svmulh_u64_x(pg,z6,b);z7=svmulh_u64_x(pg,z7,b);
            BARRIER8(z0,z1,z2,z3,z4,z5,z6,z7);
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/(ITERS*8);
        printf("  svmulh x8 (alone)        %5.3f cyc/inst\n", per);
    }

    // 3. svmul + svmulh, SAME operands (8 pairs)
    {
        svuint64_t z0=a,z1=a,z2=a,z3=a,z4=a,z5=a,z6=a,z7=a;
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            z0=svmul_u64_x(pg,z0,b); z0=svmulh_u64_x(pg,z0,b);
            z1=svmul_u64_x(pg,z1,b); z1=svmulh_u64_x(pg,z1,b);
            z2=svmul_u64_x(pg,z2,b); z2=svmulh_u64_x(pg,z2,b);
            z3=svmul_u64_x(pg,z3,b); z3=svmulh_u64_x(pg,z3,b);
            BARRIER8(z0,z1,z2,z3,z4,z5,z6,z7);
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/(ITERS*8*2);
        printf("  svmul+svmulh same op     %5.3f cyc/inst  (fused → ~0.5)\n", per);
    }

    // 4. svmul + svmulh, DIFFERENT operands (8 pairs)
    {
        svuint64_t z0=a,z1=a,z2=a,z3=a,z4=a,z5=a,z6=a,z7=a;
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            z0=svmul_u64_x(pg,z0,b); z0=svmulh_u64_x(pg,z0,c);
            z1=svmul_u64_x(pg,z1,b); z1=svmulh_u64_x(pg,z1,c);
            z2=svmul_u64_x(pg,z2,b); z2=svmulh_u64_x(pg,z2,c);
            z3=svmul_u64_x(pg,z3,b); z3=svmulh_u64_x(pg,z3,c);
            BARRIER8(z0,z1,z2,z3,z4,z5,z6,z7);
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/(ITERS*8*2);
        printf("  svmul+svmulh diff op     %5.3f cyc/inst  (no fuse → ~1.0)\n", per);
    }

    // 5. Full SVE Barrett: svmul + svmulh + svmls (latency, chained)
    {
        svuint64_t z = a;
        svuint64_t sv_op = svdup_n_u64(0x123456789ULL);
        svuint64_t sv_q  = svdup_n_u64(0x123456789ULL);  // same as op → fusion
        svuint64_t sv_p  = svdup_n_u64(0xFFFFFFFFFFFFFFC5ULL);
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            svuint64_t hi = svmulh_u64_x(pg, z, sv_q);
            svuint64_t lo = svmul_u64_x(pg, z, sv_op);
            z = svmls_u64_x(pg, lo, sv_p, hi);
            __asm__ volatile(""::"w"(z));
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/ITERS;
        printf("\n  SVE Barrett op=q (lat)   %5.2f cyc\n", per);
    }

    // 6. Full SVE Barrett with op≠q (latency, chained)
    {
        svuint64_t z = a;
        svuint64_t sv_op = svdup_n_u64(0x123456789ULL);
        svuint64_t sv_q  = svdup_n_u64(0x987654321ULL);  // different from op
        svuint64_t sv_p  = svdup_n_u64(0xFFFFFFFFFFFFFFC5ULL);
        t0=cntvct();
        for(int i=0;i<ITERS;i++){
            svuint64_t hi = svmulh_u64_x(pg, z, sv_q);
            svuint64_t lo = svmul_u64_x(pg, z, sv_op);
            z = svmls_u64_x(pg, lo, sv_p, hi);
            __asm__ volatile(""::"w"(z));
        }
        t1=cntvct();
        per=(double)(t1-t0)*23.0/ITERS;
        printf("  SVE Barrett op≠q (lat)   %5.2f cyc\n", per);
    }

    return 0;
}
