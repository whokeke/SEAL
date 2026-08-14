#ifndef SEAL_UTIL_DWTHANDLER_SVE_NEW_H
#define SEAL_UTIL_DWTHANDLER_SVE_NEW_H

#include <arm_sve.h>
#include <cstdint>
#include "seal/util/defines.h"
#include "seal/util/uintarithsmallmod.h"

namespace seal
{
    namespace util
    {
        // ============================================================
        // SVE intrinsic helpers for DWT butterfly
        // ============================================================
        // Uses svmulh_u64_x (SVE2) for the 64x64->high-64 multiply; no
        // 32-bit decomposition is needed (unlike the AVX-512 variant).

        // 64x64 -> high 64 bits (native SVE2).
        static inline svuint64_t sv_mulhi_u64(svbool_t pg, svuint64_t a, svuint64_t b)
        {
            return svmulh_u64_x(pg, a, b);
        }

        // Barrett lazy modular multiply.
        // Returns: x * y.operand - tmp1 * p  (lazy, result in [0, 4p))
        // where tmp1 = high64(x * y.quotient).
        static inline svuint64_t sv_mulmod_lazy_barrett(
            svbool_t pg, svuint64_t x, uint64_t y_operand, uint64_t y_quotient, uint64_t p)
        {
            svuint64_t y_op = svdup_n_u64(y_operand);
            svuint64_t y_q = svdup_n_u64(y_quotient);
            svuint64_t pv = svdup_n_u64(p);

            // tmp1 = high64(x * y_quotient)
            svuint64_t tmp1 = svmulh_u64_x(pg, x, y_q);

            // result = x * y_operand - tmp1 * p
            svuint64_t lo = svmul_u64_x(pg, x, y_op);
            svuint64_t hi = svmul_u64_x(pg, tmp1, pv);
            return svsub_u64_x(pg, lo, hi);
        }

        // Guard: reduce from [0, 4p) to [0, 2p).  if (a >= 2p) a -= 2p
        static inline svuint64_t sv_guard(svbool_t pg, svuint64_t a, uint64_t two_p)
        {
            svuint64_t tpv = svdup_n_u64(two_p);
            svbool_t ge = svcmpge_u64(pg, a, tpv);
            return svsel_u64(ge, svsub_u64_x(pg, a, tpv), a);
        }

        // Butterfly: x = guard(x) + v; y = guard(x) + 2p - v  (lazy add/sub)
        // Input: x in [0, 4p), v = mulmod_lazy(y_val, root) in [0, 4p)
        // Output: x_out = guard(x) + v, y_out = guard(x) + 2p - v
        static inline void sv_butterfly(
            svbool_t pg, svuint64_t x, svuint64_t v, uint64_t two_p,
            svuint64_t &x_out, svuint64_t &y_out)
        {
            svuint64_t g = sv_guard(pg, x, two_p);
            svuint64_t tpv = svdup_n_u64(two_p);
            x_out = svadd_u64_x(pg, g, v);
            y_out = svsub_u64_x(pg, svadd_u64_x(pg, g, tpv), v);
        }

        // ============================================================
        // SVE DWTHandler specialization
        // ============================================================
        // Processes svcntd() elements per iteration using SVE intrinsics.
        // Predicate-based tail handling (svwhilelt_b64) removes the need for a
        // scalar tail inside the vectorized loop. Falls back to scalar when
        // gap < vector length.

        template <typename Arithmetic_ = Arithmetic<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>>
        class DWTHandlerSVE
        {
        public:
            DWTHandlerSVE() : two_p_(0), p_(0) {}

            DWTHandlerSVE(std::uint64_t modulus_value, std::uint64_t two_times_modulus)
                : arith_(Modulus(modulus_value)), two_p_(two_times_modulus),
                  p_(modulus_value) {}

            DWTHandlerSVE(const Arithmetic_ &arith) : arith_(arith), two_p_(0), p_(0) {}

            void transform_to_rev(
                std::uint64_t *values, int log_n,
                const MultiplyUIntModOperand *roots,
                const MultiplyUIntModOperand *scalar = nullptr) const
            {
                size_t n = size_t(1) << log_n;
                std::size_t gap = n >> 1;
                std::size_t m = 1;
                const auto &a = arith_;
                uint64_t two_p = two_p_;
                uint64_t p = p_;
                std::size_t vl = svcntd();

                for (; m < (n >> 1); m <<= 1)
                {
                    std::size_t offset = 0;
                    if (gap < vl)
                    {
                        // Scalar fallback (gap < vector length)
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::uint64_t u = a.guard(*x);
                                std::uint64_t v = a.mul_root(*y, r);
                                *x++ = a.add(u, v);
                                *y++ = a.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
                    else
                    {
                        // SVE vectorized loop; predicate handles the partial tail.
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;

                            for (std::size_t j = 0; j < gap; j += vl)
                            {
                                svbool_t pg = svwhilelt_b64((int64_t)j, (int64_t)gap);
                                svuint64_t vx = svld1_u64(pg, x + j);
                                svuint64_t vy = svld1_u64(pg, y + j);

                                // v = mulmod_lazy(y_val, root)
                                svuint64_t vv = sv_mulmod_lazy_barrett(pg, vy, r.operand, r.quotient, p);

                                // butterfly: x_out = guard(x) + v, y_out = guard(x) + 2p - v
                                svuint64_t x_out, y_out;
                                sv_butterfly(pg, vx, vv, two_p, x_out, y_out);

                                svst1_u64(pg, x + j, x_out);
                                svst1_u64(pg, y + j, y_out);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap >>= 1;
                }

                // Final stage (m == n/2, gap == 1)
                if (scalar != nullptr)
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        MultiplyUIntModOperand r = *++roots;
                        MultiplyUIntModOperand scaled_r = a.mul_root_scalar(r, *scalar);
                        std::uint64_t u = a.mul_scalar(a.guard(values[0]), *scalar);
                        std::uint64_t v = a.mul_root(values[1], scaled_r);
                        values[0] = a.add(u, v);
                        values[1] = a.sub(u, v);
                        values += 2;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        MultiplyUIntModOperand r = *++roots;
                        std::uint64_t u = a.guard(values[0]);
                        std::uint64_t v = a.mul_root(values[1], r);
                        values[0] = a.add(u, v);
                        values[1] = a.sub(u, v);
                        values += 2;
                    }
                }
            }

            void transform_from_rev(
                std::uint64_t *values, int log_n,
                const MultiplyUIntModOperand *roots,
                const MultiplyUIntModOperand *scalar = nullptr) const
            {
                size_t n = size_t(1) << log_n;
                std::size_t gap = 1;
                std::size_t m = n >> 1;
                const auto &a = arith_;
                uint64_t two_p = two_p_;
                uint64_t p = p_;
                std::size_t vl = svcntd();

                for (; m > 1; m >>= 1)
                {
                    std::size_t offset = 0;
                    if (gap < vl)
                    {
                        // Scalar fallback
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::uint64_t u = *x;
                                std::uint64_t v = *y;
                                *x++ = a.guard(a.add(u, v));
                                *y++ = a.mul_root(a.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    else
                    {
                        // SVE vectorized loop
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;

                            for (std::size_t j = 0; j < gap; j += vl)
                            {
                                svbool_t pg = svwhilelt_b64((int64_t)j, (int64_t)gap);
                                svuint64_t vx = svld1_u64(pg, x + j);
                                svuint64_t vy = svld1_u64(pg, y + j);

                                // x_out = guard(x + y)
                                svuint64_t sum = svadd_u64_x(pg, vx, vy);
                                svuint64_t x_out = sv_guard(pg, sum, two_p);

                                // y_out = mulmod_lazy(x - y + 2p, root)
                                // sub: x - y + 2p (lazy sub)
                                svuint64_t diff = svsub_u64_x(pg, svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vy);
                                svuint64_t y_out = sv_mulmod_lazy_barrett(pg, diff, r.operand, r.quotient, p);

                                svst1_u64(pg, x + j, x_out);
                                svst1_u64(pg, y + j, y_out);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap <<= 1;
                }

                // Final stage (gap == n/2, m == 1)
                if (scalar != nullptr)
                {
                    MultiplyUIntModOperand r = *++roots;
                    MultiplyUIntModOperand scaled_r = a.mul_root_scalar(r, *scalar);
                    std::uint64_t *x = values;
                    std::uint64_t *y = x + gap;

                    if (gap >= vl)
                    {
                        for (std::size_t j = 0; j < gap; j += vl)
                        {
                            svbool_t pg = svwhilelt_b64((int64_t)j, (int64_t)gap);
                            svuint64_t vx = svld1_u64(pg, x + j);
                            svuint64_t vy = svld1_u64(pg, y + j);

                            svuint64_t g = sv_guard(pg, vx, two_p);
                            svuint64_t sum = svadd_u64_x(pg, g, vy);
                            svuint64_t x_out = sv_guard(pg, sum, two_p);
                            svuint64_t diff = svsub_u64_x(pg, svadd_u64_x(pg, g, svdup_n_u64(two_p)), vy);
                            svuint64_t y_out = sv_mulmod_lazy_barrett(pg, diff, scaled_r.operand, scaled_r.quotient, p);
                            // Also need mul_scalar for x_out
                            svuint64_t x_final = sv_mulmod_lazy_barrett(pg, x_out, scalar->operand, scalar->quotient, p);
                            svst1_u64(pg, x + j, x_final);
                            svst1_u64(pg, y + j, y_out);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            std::uint64_t u = a.guard(x[j]);
                            std::uint64_t v = y[j];
                            x[j] = a.mul_scalar(a.guard(a.add(u, v)), *scalar);
                            y[j] = a.mul_root(a.sub(u, v), scaled_r);
                        }
                    }
                }
                else
                {
                    MultiplyUIntModOperand r = *++roots;
                    std::uint64_t *x = values;
                    std::uint64_t *y = x + gap;

                    if (gap >= vl)
                    {
                        for (std::size_t j = 0; j < gap; j += vl)
                        {
                            svbool_t pg = svwhilelt_b64((int64_t)j, (int64_t)gap);
                            svuint64_t vx = svld1_u64(pg, x + j);
                            svuint64_t vy = svld1_u64(pg, y + j);
                            svuint64_t sum = svadd_u64_x(pg, vx, vy);
                            svuint64_t x_out = sv_guard(pg, sum, two_p);
                            svuint64_t diff = svsub_u64_x(pg, svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vy);
                            svuint64_t y_out = sv_mulmod_lazy_barrett(pg, diff, r.operand, r.quotient, p);
                            svst1_u64(pg, x + j, x_out);
                            svst1_u64(pg, y + j, y_out);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            std::uint64_t u = x[j];
                            std::uint64_t v = y[j];
                            x[j] = a.guard(a.add(u, v));
                            y[j] = a.mul_root(a.sub(u, v), r);
                        }
                    }
                }
            }

        private:
            Arithmetic_ arith_;
            uint64_t two_p_;
            uint64_t p_;
        };
    } // namespace util
} // namespace seal

#endif // SEAL_UTIL_DWTHANDLER_SVE_NEW_H
