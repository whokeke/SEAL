#ifndef SEAL_UTIL_DWTHANDLER_AVX512_H
#define SEAL_UTIL_DWTHANDLER_AVX512_H

#include <immintrin.h>
#include <cstdint>
#include "seal/util/defines.h"
#include "seal/util/uintarithsmallmod.h"

namespace seal
{
    namespace util
    {
        // ============================================================
        // AVX-512 intrinsic helpers for DWT butterfly (Zen5 target)
        // ============================================================

        // 64x64 -> high 64 bits, using 32-bit partial products + vector carry chain.
        // No scalar mulx needed; all ops are vpmullq/vpaddq/shift/and/or.
        static inline __m512i _mm512_mulhi_epi64(__m512i a, __m512i b)
        {
            __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFF);

            __m512i a_hi = _mm512_srli_epi64(a, 32);
            __m512i a_lo = _mm512_and_si512(a, mask32);
            __m512i b_hi = _mm512_srli_epi64(b, 32);
            __m512i b_lo = _mm512_and_si512(b, mask32);

            // 4 partial products (32x32->64, all fit in 64-bit lanes)
            __m512i LL = _mm512_mullo_epi64(a_lo, b_lo);
            __m512i ML = _mm512_mullo_epi64(a_lo, b_hi);
            __m512i MH = _mm512_mullo_epi64(a_hi, b_lo);
            __m512i HH = _mm512_mullo_epi64(a_hi, b_hi);

            // carry chain 1: sum1 = LL_hi + ML_lo + MH_lo
            // carry1 = sum1 >> 32 (value is 0, 1, or 2)
            __m512i sum1 = _mm512_add_epi64(_mm512_srli_epi64(LL, 32),
                                            _mm512_and_si512(ML, mask32));
            sum1 = _mm512_add_epi64(sum1, _mm512_and_si512(MH, mask32));
            __m512i carry1 = _mm512_srli_epi64(sum1, 32);

            // carry chain 2: sum2 = HH_lo + ML_hi + MH_hi + carry1
            // carry2 = sum2 >> 32 (value is 0, 1, or 2)
            __m512i sum2 = _mm512_add_epi64(_mm512_and_si512(HH, mask32),
                                            _mm512_srli_epi64(ML, 32));
            sum2 = _mm512_add_epi64(sum2, _mm512_srli_epi64(MH, 32));
            sum2 = _mm512_add_epi64(sum2, carry1);
            __m512i carry2 = _mm512_srli_epi64(sum2, 32);

            // high64 = (HH_hi + carry2) << 32 | (sum2 & mask32)
            __m512i hi = _mm512_add_epi64(_mm512_srli_epi64(HH, 32), carry2);
            return _mm512_or_si512(_mm512_slli_epi64(hi, 32),
                                   _mm512_and_si512(sum2, mask32));
        }

        // Barrett lazy modular multiply for 8 elements (AVX-512)
        // Returns: x * y.operand - tmp1 * p  (lazy, result in [0, 4p))
        // where tmp1 = high64(x * y.quotient)
        static inline __m512i mm512_mulmod_lazy_barrett(
            __m512i x, uint64_t y_operand, uint64_t y_quotient, uint64_t p)
        {
            __m512i y_op = _mm512_set1_epi64(y_operand);
            __m512i y_q  = _mm512_set1_epi64(y_quotient);
            __m512i pv  = _mm512_set1_epi64(p);

            // tmp1 = high64(x * y_quotient)  — using our mulhi
            __m512i tmp1 = _mm512_mulhi_epi64(x, y_q);

            // result = x * y_operand - tmp1 * p
            __m512i lo = _mm512_mullo_epi64(x, y_op);
            __m512i hi = _mm512_mullo_epi64(tmp1, pv);
            return _mm512_sub_epi64(lo, hi);
        }

        // Guard: reduce from [0, 4p) to [0, 2p)
        // if (a >= 2p) a -= 2p
        static inline __m512i mm512_guard(__m512i a, uint64_t two_p)
        {
            __m512i tpv = _mm512_set1_epi64(two_p);
            __mmask8 ge = _mm512_cmpge_epi64_mask(a, tpv);
            return _mm512_mask_sub_epi64(a, ge, a, tpv);
        }

        // Butterfly: x = guard(x) + v; y = guard(x) + 2p - v  (lazy add/sub)
        // Input: x in [0, 2p), v = mulmod_lazy(y_val, root) in [0, 4p)
        // Output: x_out = guard(x) + v, y_out = guard(x) + 2p - v
        static inline void mm512_butterfly(
            __m512i x, __m512i v, uint64_t two_p,
            __m512i &x_out, __m512i &y_out)
        {
            __m512i tpv = _mm512_set1_epi64(two_p);
            __m512i g = mm512_guard(x, two_p);
            x_out = _mm512_add_epi64(g, v);
            y_out = _mm512_sub_epi64(_mm512_add_epi64(g, tpv), v);
        }

        // ============================================================
        // AVX-512 DWTHandler specialization
        // ============================================================
        // Processes 8 elements per iteration using AVX-512 intrinsics.
        // Requires gap >= 8 and gap % 8 == 0 for the vectorized loop.
        // Falls back to scalar for gap < 8.

        template <typename Arithmetic_ = Arithmetic<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>>
        class DWTHandlerAVX512
        {
        public:
            DWTHandlerAVX512() {}
            DWTHandlerAVX512(uint64_t modulus_value, uint64_t two_times_modulus)
                : arith_(Modulus(modulus_value)), two_p_(two_times_modulus),
                  p_(modulus_value) {}
            DWTHandlerAVX512(const Arithmetic_ &arith)
                : arith_(arith), two_p_(0), p_(0) {
                // Get values through public guard interface
                // two_p_ = arith.two_times_modulus_ (private)
                // p_ = arith.modulus_.value() (private)
                // Workaround: compute from guard
            }

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

                for (; m < (n >> 1); m <<= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 8)
                    {
                        // Scalar fallback (gap 1..7)
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
                        // AVX-512 vectorized loop (gap >= 8)
                        std::size_t vec_count = gap & ~static_cast<std::size_t>(7); // floor(gap/8)*8
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;

                            // Vectorized: process 8 butterflies per iter
                            for (std::size_t j = 0; j < vec_count; j += 8)
                            {
                                __m512i vx = _mm512_loadu_si512((__m512i *)(x + j));
                                __m512i vy = _mm512_loadu_si512((__m512i *)(y + j));

                                // guard x
                                __m512i gx = mm512_guard(vx, two_p);

                                // v = mulmod_lazy(y_val, root)
                                __m512i vv = mm512_mulmod_lazy_barrett(
                                    vy, r.operand, r.quotient, p);

                                // butterfly: x_out = gx + vv, y_out = gx + 2p - vv
                                __m512i x_out, y_out;
                                mm512_butterfly(vx, vv, two_p, x_out, y_out);
                                // Note: butterfly already guards x, but we already did mm512_guard above.
                                // Let's inline it directly:
                                x_out = _mm512_add_epi64(gx, vv);
                                y_out = _mm512_sub_epi64(
                                    _mm512_add_epi64(gx, _mm512_set1_epi64(two_p)), vv);

                                _mm512_storeu_si512((__m512i *)(x + j), x_out);
                                _mm512_storeu_si512((__m512i *)(y + j), y_out);
                            }

                            // Scalar tail (gap % 8 remaining)
                            std::uint64_t *xs = x + vec_count;
                            std::uint64_t *ys = y + vec_count;
                            for (std::size_t j = vec_count; j < gap; j++)
                            {
                                std::uint64_t u = a.guard(*xs);
                                std::uint64_t v = a.mul_root(*ys, r);
                                *xs++ = a.add(u, v);
                                *ys++ = a.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap >>= 1;
                }

                // Final stage (m == n/2, gap == 1)
                if (scalar != nullptr)
                {
                    MultiplyUIntModOperand r;
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
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

                for (; m > 1; m >>= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 8)
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
                        // AVX-512 vectorized loop
                        std::size_t vec_count = gap & ~static_cast<std::size_t>(7);
                        for (std::size_t i = 0; i < m; i++)
                        {
                            MultiplyUIntModOperand r = *++roots;
                            std::uint64_t *x = values + offset;
                            std::uint64_t *y = x + gap;

                            for (std::size_t j = 0; j < vec_count; j += 8)
                            {
                                __m512i vx = _mm512_loadu_si512((__m512i *)(x + j));
                                __m512i vy = _mm512_loadu_si512((__m512i *)(y + j));

                                // x_out = guard(x + y)
                                __m512i sum = _mm512_add_epi64(vx, vy);
                                __m512i x_out = mm512_guard(sum, two_p);

                                // y_out = mulmod_lazy(x - y + 2p, root)
                                // sub: x - y + 2p (lazy sub)
                                __m512i diff = _mm512_sub_epi64(
                                    _mm512_add_epi64(vx, _mm512_set1_epi64(two_p)), vy);
                                __m512i y_out = mm512_mulmod_lazy_barrett(
                                    diff, r.operand, r.quotient, p);

                                _mm512_storeu_si512((__m512i *)(x + j), x_out);
                                _mm512_storeu_si512((__m512i *)(y + j), y_out);
                            }

                            // Scalar tail
                            std::uint64_t *xs = x + vec_count;
                            std::uint64_t *ys = y + vec_count;
                            for (std::size_t j = vec_count; j < gap; j++)
                            {
                                std::uint64_t u = *xs;
                                std::uint64_t v = *ys;
                                *xs++ = a.guard(a.add(u, v));
                                *ys++ = a.mul_root(a.sub(u, v), r);
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

                    if (gap >= 8)
                    {
                        std::size_t vec_count = gap & ~static_cast<std::size_t>(7);
                        for (std::size_t j = 0; j < vec_count; j += 8)
                        {
                            __m512i vx = _mm512_loadu_si512((__m512i *)(x + j));
                            __m512i vy = _mm512_loadu_si512((__m512i *)(y + j));
                            __m512i g = mm512_guard(vx, two_p);
                            __m512i sum = _mm512_add_epi64(g, vy);
                            __m512i x_out = mm512_guard(sum, two_p);
                            __m512i diff = _mm512_sub_epi64(
                                _mm512_add_epi64(g, _mm512_set1_epi64(two_p)), vy);
                            __m512i y_out = mm512_mulmod_lazy_barrett(
                                diff, scaled_r.operand, scaled_r.quotient, p);
                            // Also need mul_scalar for x_out
                            __m512i x_final = mm512_mulmod_lazy_barrett(
                                x_out, scalar->operand, scalar->quotient, p);
                            _mm512_storeu_si512((__m512i *)(x + j), x_final);
                            _mm512_storeu_si512((__m512i *)(y + j), y_out);
                        }
                        // Scalar tail
                        for (std::size_t j = vec_count; j < gap; j++)
                        {
                            std::uint64_t u = a.guard(x[j]);
                            std::uint64_t v = y[j];
                            x[j] = a.mul_scalar(a.guard(a.add(u, v)), *scalar);
                            y[j] = a.mul_root(a.sub(u, v), scaled_r);
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

                    if (gap >= 8)
                    {
                        std::size_t vec_count = gap & ~static_cast<std::size_t>(7);
                        for (std::size_t j = 0; j < vec_count; j += 8)
                        {
                            __m512i vx = _mm512_loadu_si512((__m512i *)(x + j));
                            __m512i vy = _mm512_loadu_si512((__m512i *)(y + j));
                            __m512i sum = _mm512_add_epi64(vx, vy);
                            __m512i x_out = mm512_guard(sum, two_p);
                            __m512i diff = _mm512_sub_epi64(
                                _mm512_add_epi64(vx, _mm512_set1_epi64(two_p)), vy);
                            __m512i y_out = mm512_mulmod_lazy_barrett(
                                diff, r.operand, r.quotient, p);
                            _mm512_storeu_si512((__m512i *)(x + j), x_out);
                            _mm512_storeu_si512((__m512i *)(y + j), y_out);
                        }
                        for (std::size_t j = vec_count; j < gap; j++)
                        {
                            std::uint64_t u = x[j];
                            std::uint64_t v = y[j];
                            x[j] = a.guard(a.add(u, v));
                            y[j] = a.mul_root(a.sub(u, v), r);
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

#endif // SEAL_UTIL_DWTHANDLER_AVX512_H
