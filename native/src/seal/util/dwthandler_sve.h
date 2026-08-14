// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/pointer.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <stdexcept>
#include <type_traits>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace seal
{
    namespace util
    {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        // AVX-512 has no vpmulhq for 64-bit integers.
        // Emulate _mm512_mulhi_epu64 by decomposing into 32-bit halves.
        // a*b = p_hh*2^64 + (p_hl+p_lh)*2^32 + p_ll
        // high64 = p_hh + (mid>>32) + (mid_carry << 32) + carry_from(p_ll + mid_lo<<32)
        static inline __m512i _mm512_mulhi_epu64(__m512i a, __m512i b)
        {
            __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFF);
            __m512i a_lo = _mm512_and_si512(a, mask32);
            __m512i a_hi = _mm512_srli_epi64(a, 32);
            __m512i b_lo = _mm512_and_si512(b, mask32);
            __m512i b_hi = _mm512_srli_epi64(b, 32);

            // 32x32->64 partial products via vpmuludq
            __m512i p_hh = _mm512_mul_epu32(a_hi, b_hi);
            __m512i p_hl = _mm512_mul_epu32(a_hi, b_lo);
            __m512i p_lh = _mm512_mul_epu32(a_lo, b_hi);
            __m512i p_ll = _mm512_mul_epu32(a_lo, b_lo);

            // mid = p_hl + p_lh (64-bit, may overflow into bit 64)
            __m512i mid = _mm512_add_epi64(p_hl, p_lh);
            __mmask8 mid_carry = _mm512_cmplt_epu64_mask(mid, p_hl);

            // mid_lo contributes to the low 64 bits: t = p_ll + (mid_lo << 32)
            __m512i mid_lo = _mm512_and_si512(mid, mask32);
            __m512i mid_lo_shifted = _mm512_slli_epi64(mid_lo, 32);
            __m512i t = _mm512_add_epi64(p_ll, mid_lo_shifted);
            __mmask8 t_carry = _mm512_cmplt_epu64_mask(t, p_ll);

            // mid_hi = mid >> 32 (bits [63:32] of the wrapped sum)
            __m512i mid_hi = _mm512_srli_epi64(mid, 32);

            // high64 = p_hh + mid_hi + (mid_carry << 32) + t_carry
            __m512i hi = _mm512_add_epi64(p_hh, mid_hi);
            hi = _mm512_add_epi64(hi,
                _mm512_slli_epi64(_mm512_maskz_set1_epi64(mid_carry, 1), 32));
            hi = _mm512_add_epi64(hi,
                _mm512_maskz_set1_epi64(t_carry, 1));

            return hi;
        }

        // Barrett lazy: result = x * y_op - high64(x * y_q) * p
        static inline __m512i _mm512_mulmod_lazy(
            __m512i x, uint64_t y_op, uint64_t y_q, uint64_t p)
        {
            __m512i tmp1 = _mm512_mulhi_epu64(x, _mm512_set1_epi64(y_q));
            __m512i lo = _mm512_mullo_epi64(x, _mm512_set1_epi64(y_op));
            __m512i hi = _mm512_mullo_epi64(tmp1, _mm512_set1_epi64(p));
            return _mm512_sub_epi64(lo, hi);
        }

        // Guard: if (a >= 2p) a -= 2p
        static inline __m512i _mm512_guard(__m512i a, uint64_t two_p)
        {
            __mmask8 ge = _mm512_cmpge_epu64_mask(a, _mm512_set1_epi64(two_p));
            return _mm512_mask_sub_epi64(a, ge, a, _mm512_set1_epi64(two_p));
        }

        // Aligned load/store: SEAL allocations are 64-byte aligned
        static inline __m512i _mm512_load_aligned(const void *p)
        {
            return _mm512_load_si512(static_cast<const __m512i *>(__builtin_assume_aligned(p, 64)));
        }
        static inline void _mm512_store_aligned(void *p, __m512i v)
        {
            _mm512_store_si512(static_cast<__m512i *>(__builtin_assume_aligned(p, 64)), v);
        }
#endif

#ifdef __ARM_FEATURE_SVE
        // SVE helper: lazy Barrett modular multiply, result = x*y_op - p*umulh(x,y_q) ∈ [0, p)
        static inline svuint64_t sve_mulmod_lazy(
            svuint64_t x, std::uint64_t y_op, std::uint64_t y_q, std::uint64_t p, svbool_t pg)
        {
            svuint64_t hi = svmulh_u64_x(pg, x, svdup_n_u64(y_q));
            svuint64_t lo = svmul_u64_x(pg, x, svdup_n_u64(y_op));
            return svmls_u64_x(pg, lo, svdup_n_u64(p), hi);
        }

        // SVE helper: guard to [0, 2p) by subtracting 2p if >= 2p
        static inline svuint64_t sve_guard(svuint64_t a, std::uint64_t two_p, svbool_t pg)
        {
            svbool_t ge = svcmpge_u64(pg, a, svdup_n_u64(two_p));
            return svsub_u64_m(ge, a, svdup_n_u64(two_p));
        }
#endif

        /**
        Provides an interface to all necessary arithmetic of the number structure that specializes a DWTHandler.
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        class Arithmetic
        {
        public:
            ValueType add(const ValueType &a, const ValueType &b) const;
            ValueType sub(const ValueType &a, const ValueType &b) const;
            ValueType mul_root(const ValueType &a, const RootType &r) const;
            ValueType mul_scalar(const ValueType &a, const ScalarType &s) const;
            RootType mul_root_scalar(const RootType &r, const ScalarType &s) const;
            ValueType guard(const ValueType &a) const;
        };

        /**
        Provides an interface that performs the fast discrete weighted transform (DWT) and its inverse that are used to
        accelerate polynomial multiplications, batch multiple messages into a single plaintext polynomial. This class
        template is specialized with integer modular arithmetic for DWT over integer quotient rings, and is used in
        polynomial multiplications and BatchEncoder. It is also specialized with double-precision complex arithmetic for
        DWT over the complex field, which is used in CKKSEncoder.
        */
        template <typename ValueType, typename RootType, typename ScalarType>
        class DWTHandler
        {
        public:
            DWTHandler()
            {}

            DWTHandler(const Arithmetic<ValueType, RootType, ScalarType> &num_struct) : arithmetic_(num_struct)
            {}

            void transform_to_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                size_t n = size_t(1) << log_n;
                RootType r;
                ValueType u;
                ValueType v;
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                std::size_t gap = n >> 1;
                std::size_t m = 1;

                for (; m < (n >> 1); m <<= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 4)
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = arithmetic_.guard(*x);
                                v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v);
                                *y++ = arithmetic_.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                    else if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                std::size_t j = 0;
                                for (; j + 8 <= gap; j += 8)
                                {
                                    __m512i vx = _mm512_load_aligned(x + j);
                                    __m512i vy = _mm512_load_aligned(y + j);
                                    __m512i gx = _mm512_guard(vx, two_p);
                                    __m512i vv = _mm512_mulmod_lazy(vy, r.operand, r.quotient, p);
                                    __m512i xo = _mm512_add_epi64(gx, vv);
                                    __m512i yo = _mm512_sub_epi64(
                                        _mm512_add_epi64(gx, _mm512_set1_epi64(two_p)), vv);
                                    _mm512_store_aligned(x + j, xo);
                                    _mm512_store_aligned(y + j, yo);
                                }
                                for (; j < gap; j++)
                                {
                                    u = arithmetic_.guard(x[j]);
                                    v = arithmetic_.mul_root(y[j], r);
                                    x[j] = arithmetic_.add(u, v);
                                    y[j] = arithmetic_.sub(u, v);
                                }
                                offset += gap << 1;
                            }
                        }
                        else
                        {
                            // gap is always a power of 2 >= 4 here, so gap % 4 == 0
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                                {
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                }
                                offset += gap << 1;
                            }
                        }
                    }
#elif defined(__ARM_FEATURE_SVE)
                    else if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const uint64_t N = svcntd();
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                std::size_t j = 0;
                                for (; j + N <= gap; j += N)
                                {
                                    svuint64_t vx = svld1_u64(svptrue_b64(), x + j);
                                    svuint64_t vy = svld1_u64(svptrue_b64(), y + j);
                                    vx = sve_guard(vx, two_p, svptrue_b64());
                                    svuint64_t vv = sve_mulmod_lazy(vy, r.operand, r.quotient, p, svptrue_b64());
                                    svuint64_t xo = svadd_u64_x(svptrue_b64(), vx, vv);
                                    svuint64_t yo = svsub_u64_x(svptrue_b64(),
                                        svadd_u64_x(svptrue_b64(), vx, svdup_n_u64(two_p)), vv);
                                    svst1_u64(svptrue_b64(), x + j, xo);
                                    svst1_u64(svptrue_b64(), y + j, yo);
                                }
                                while (j < gap)
                                {
                                    svbool_t pg = svwhilelt_b64(j, gap);
                                    svuint64_t vx = svld1_u64(pg, x + j);
                                    svuint64_t vy = svld1_u64(pg, y + j);
                                    vx = sve_guard(vx, two_p, pg);
                                    svuint64_t vv = sve_mulmod_lazy(vy, r.operand, r.quotient, p, pg);
                                    svuint64_t xo = svadd_u64_x(pg, vx, vv);
                                    svuint64_t yo = svsub_u64_x(pg,
                                        svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vv);
                                    svst1_u64(pg, x + j, xo);
                                    svst1_u64(pg, y + j, yo);
                                    j += N;
                                }
                                offset += gap << 1;
                            }
                        }
                        else
                        {
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                for (std::size_t j = 0; j < gap; j += 4)
                                {
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                    u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                    *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                }
                                offset += gap << 1;
                            }
                        }
                    }
#endif
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                                u = arithmetic_.guard(*x); v = arithmetic_.mul_root(*y, r);
                                *x++ = arithmetic_.add(u, v); *y++ = arithmetic_.sub(u, v);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap >>= 1;
                }

                if (scalar != nullptr)
                {
                    RootType scaled_r;
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                        u = arithmetic_.mul_scalar(arithmetic_.guard(values[0]), *scalar);
                        v = arithmetic_.mul_root(values[1], scaled_r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
                }
                else
                {
#ifdef __ARM_FEATURE_SVE
                    if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        const std::uint64_t p = arithmetic_.modulus_.value();
                        const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                        const svuint64_t sv_p = svdup_n_u64(p);
                        const svuint64_t sv_two_p = svdup_n_u64(two_p);
                        const uint64_t N = svcntd();
                        const RootType *r_ptr = roots + 1;
                        ValueType *v_ptr = values;

                        std::size_t i = 0;
                        while (i < m)
                        {
                            svbool_t pg = svwhilelt_b64(i, m);

                            // Gather root operands and quotients
                            // RootType = MultiplyUIntModOperand = {operand, quotient}, stride 2
                            svuint64_t idx_op = svindex_u64(0, sizeof(RootType) / sizeof(std::uint64_t));
                            svuint64_t sv_r_op = svld1_gather_u64index_u64(pg,
                                reinterpret_cast<const std::uint64_t *>(r_ptr), idx_op);
                            svuint64_t idx_q = svindex_u64(1, sizeof(RootType) / sizeof(std::uint64_t));
                            svuint64_t sv_r_q = svld1_gather_u64index_u64(pg,
                                reinterpret_cast<const std::uint64_t *>(r_ptr), idx_q);

                            // Load value pairs with deinterleave
                            svuint64x2_t pair = svld2_u64(pg,
                                reinterpret_cast<const std::uint64_t *>(v_ptr));
                            svuint64_t va = svget2_u64(pair, 0);
                            svuint64_t vb = svget2_u64(pair, 1);

                            // Guard va → [0, 2p)
                            va = sve_guard(va, two_p, pg);

                            // Lazy Barrett: vv = vb * r.operand - p * hi(vb, r.quotient)
                            svuint64_t v_hi = svmulh_u64_x(pg, vb, sv_r_q);
                            svuint64_t v_lo = svmul_u64_x(pg, vb, sv_r_op);
                            svuint64_t vv = svmls_u64_x(pg, v_lo, sv_p, v_hi);

                            // Butterfly: xo = va + vv, yo = va + 2p - vv
                            svuint64_t xo = svadd_u64_x(pg, va, vv);
                            svuint64_t yo = svsub_u64_x(pg,
                                svadd_u64_x(pg, va, sv_two_p), vv);

                            // Store with interleave
                            svuint64x2_t out = svcreate2_u64(xo, yo);
                            svst2_u64(pg,
                                reinterpret_cast<std::uint64_t *>(v_ptr), out);

                            r_ptr += N;
                            v_ptr += 2 * N;
                            i += N;
                        }
                    }
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            u = arithmetic_.guard(values[0]);
                            v = arithmetic_.mul_root(values[1], r);
                            values[0] = arithmetic_.add(u, v);
                            values[1] = arithmetic_.sub(u, v);
                            values += 2;
                        }
                    }
#else
                    for (std::size_t i = 0; i < m; i++)
                    {
                        r = *++roots;
                        u = arithmetic_.guard(values[0]);
                        v = arithmetic_.mul_root(values[1], r);
                        values[0] = arithmetic_.add(u, v);
                        values[1] = arithmetic_.sub(u, v);
                        values += 2;
                    }
#endif
                }
            }

            void transform_from_rev(
                ValueType *values, int log_n, const RootType *roots, const ScalarType *scalar = nullptr) const
            {
                size_t n = size_t(1) << log_n;
                RootType r;
                ValueType u;
                ValueType v;
                ValueType *x = nullptr;
                ValueType *y = nullptr;
                std::size_t gap = 1;
                std::size_t m = n >> 1;

                for (; m > 1; m >>= 1)
                {
                    std::size_t offset = 0;
                    if (gap < 4)
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                    else if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                std::size_t j = 0;
                                for (; j + 8 <= gap; j += 8)
                                {
                                    __m512i vx = _mm512_load_aligned(x + j);
                                    __m512i vy = _mm512_load_aligned(y + j);
                                    __m512i sum = _mm512_add_epi64(vx, vy);
                                    __m512i xo = _mm512_guard(sum, two_p);
                                    __m512i diff = _mm512_sub_epi64(
                                        _mm512_add_epi64(vx, _mm512_set1_epi64(two_p)), vy);
                                __m512i yo = _mm512_mulmod_lazy(diff, scaled_r.operand, scaled_r.quotient, p);
                                    _mm512_store_aligned(x + j, xo);
                                    _mm512_store_aligned(y + j, yo);
                                }
                                for (; j < gap; j++)
                                {
                                    u = x[j];
                                    v = y[j];
                                    x[j] = arithmetic_.guard(arithmetic_.add(u, v));
                                    y[j] = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                }
                                offset += gap << 1;
                            }
                        }
                        else
                        {
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                                {
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                }
                                offset += gap << 1;
                            }
                        }
                    }
                    }
#elif defined(__ARM_FEATURE_SVE)
                    else if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const uint64_t N = svcntd();
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                std::size_t j = 0;
                                for (; j + N <= gap; j += N)
                                {
                                    svuint64_t vx = svld1_u64(svptrue_b64(), x + j);
                                    svuint64_t vy = svld1_u64(svptrue_b64(), y + j);
                                    svuint64_t sum = svadd_u64_x(svptrue_b64(), vx, vy);
                                    svuint64_t xo = sve_guard(sum, two_p, svptrue_b64());
                                    svuint64_t diff = svsub_u64_x(svptrue_b64(),
                                        svadd_u64_x(svptrue_b64(), vx, svdup_n_u64(two_p)), vy);
                                    svuint64_t yo = sve_mulmod_lazy(diff, r.operand, r.quotient, p, svptrue_b64());
                                    svst1_u64(svptrue_b64(), x + j, xo);
                                    svst1_u64(svptrue_b64(), y + j, yo);
                                }
                                while (j < gap)
                                {
                                    svbool_t pg = svwhilelt_b64(j, gap);
                                    svuint64_t vx = svld1_u64(pg, x + j);
                                    svuint64_t vy = svld1_u64(pg, y + j);
                                    svuint64_t sum = svadd_u64_x(pg, vx, vy);
                                    svuint64_t xo = sve_guard(sum, two_p, pg);
                                    svuint64_t diff = svsub_u64_x(pg,
                                        svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vy);
                                    svuint64_t yo = sve_mulmod_lazy(diff, r.operand, r.quotient, p, pg);
                                    svst1_u64(pg, x + j, xo);
                                    svst1_u64(pg, y + j, yo);
                                    j += N;
                                }
                                offset += gap << 1;
                            }
                        }
                        else
                        {
                            for (std::size_t i = 0; i < m; i++)
                            {
                                r = *++roots;
                                x = values + offset;
                                y = x + gap;
                                for (std::size_t j = 0; j < gap; j += 4)
                                {
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                    u = *x; v = *y;
                                    *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                    *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                }
                                offset += gap << 1;
                            }
                        }
                    }
#endif
                    else
                    {
                        for (std::size_t i = 0; i < m; i++)
                        {
                            r = *++roots;
                            x = values + offset;
                            y = x + gap;
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                            offset += gap << 1;
                        }
                    }
                    gap <<= 1;
                }

                if (scalar != nullptr)
                {
                    r = *++roots;
                    RootType scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                    x = values;
                    y = x + gap;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                    if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            std::size_t j = 0;
                            for (; j + 8 <= gap; j += 8)
                            {
                                __m512i vx = _mm512_load_aligned(x + j);
                                __m512i vy = _mm512_load_aligned(y + j);
                                __m512i gx = _mm512_guard(vx, two_p);
                                __m512i sum = _mm512_add_epi64(gx, vy);
                                __m512i gsum = _mm512_guard(sum, two_p);
                                __m512i xo = _mm512_mulmod_lazy(gsum, scalar->operand, scalar->quotient, p);
                                __m512i diff = _mm512_sub_epi64(
                                    _mm512_add_epi64(gx, _mm512_set1_epi64(two_p)), vy);
                                __m512i yo = _mm512_mulmod_lazy(diff, scaled_r.operand, scaled_r.quotient, p);
                                _mm512_store_aligned(x + j, xo);
                                _mm512_store_aligned(y + j, yo);
                            }
                            for (; j < gap; j++)
                            {
                                u = arithmetic_.guard(x[j]);
                                v = y[j];
                                x[j] = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                y[j] = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                            }
                        }
                    }
                    else
#elif defined(__ARM_FEATURE_SVE)
                    if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const uint64_t N = svcntd();
                            std::size_t j = 0;
                            for (; j + N <= gap; j += N)
                            {
                                svuint64_t vx = svld1_u64(svptrue_b64(), x + j);
                                svuint64_t vy = svld1_u64(svptrue_b64(), y + j);
                                vx = sve_guard(vx, two_p, svptrue_b64());
                                svuint64_t sum = svadd_u64_x(svptrue_b64(), vx, vy);
                                svuint64_t gsum = sve_guard(sum, two_p, svptrue_b64());
                                svuint64_t xo = sve_mulmod_lazy(gsum, scalar->operand, scalar->quotient, p, svptrue_b64());
                                svuint64_t diff = svsub_u64_x(svptrue_b64(),
                                    svadd_u64_x(svptrue_b64(), vx, svdup_n_u64(two_p)), vy);
                                svuint64_t yo = sve_mulmod_lazy(diff, scaled_r.operand, scaled_r.quotient, p, svptrue_b64());
                                svst1_u64(svptrue_b64(), x + j, xo);
                                svst1_u64(svptrue_b64(), y + j, yo);
                            }
                            while (j < gap)
                            {
                                svbool_t pg = svwhilelt_b64(j, gap);
                                svuint64_t vx = svld1_u64(pg, x + j);
                                svuint64_t vy = svld1_u64(pg, y + j);
                                vx = sve_guard(vx, two_p, pg);
                                svuint64_t sum = svadd_u64_x(pg, vx, vy);
                                svuint64_t gsum = sve_guard(sum, two_p, pg);
                                svuint64_t xo = sve_mulmod_lazy(gsum, scalar->operand, scalar->quotient, p, pg);
                                svuint64_t diff = svsub_u64_x(pg,
                                    svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vy);
                                svuint64_t yo = sve_mulmod_lazy(diff, scaled_r.operand, scaled_r.quotient, p, pg);
                                svst1_u64(pg, x + j, xo);
                                svst1_u64(pg, y + j, yo);
                                j += N;
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4)
                            {
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                            }
                        }
                    }
                    else
#endif
                    {
                        if (gap < 4)
                        {
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = arithmetic_.guard(*x);
                                v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                                u = arithmetic_.guard(*x); v = *y;
                                *x++ = arithmetic_.mul_scalar(arithmetic_.guard(arithmetic_.add(u, v)), *scalar);
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), scaled_r);
                            }
                        }
                    }
                }
                else
                {
                    r = *++roots;
                    x = values;
                    y = x + gap;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
                    if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            std::size_t j = 0;
                            for (; j + 8 <= gap; j += 8)
                            {
                                __m512i vx = _mm512_load_aligned(x + j);
                                __m512i vy = _mm512_load_aligned(y + j);
                                __m512i sum = _mm512_add_epi64(vx, vy);
                                __m512i xo = _mm512_guard(sum, two_p);
                                __m512i diff = _mm512_sub_epi64(
                                    _mm512_add_epi64(vx, _mm512_set1_epi64(two_p)), vy);
                                __m512i yo = _mm512_mulmod_lazy(diff, r.operand, r.quotient, p);
                                _mm512_store_aligned(x + j, xo);
                                _mm512_store_aligned(y + j, yo);
                            }
                            for (; j < gap; j++)
                            {
                                u = x[j];
                                v = y[j];
                                x[j] = arithmetic_.guard(arithmetic_.add(u, v));
                                y[j] = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                        }
                    }
                    else
#elif defined(__ARM_FEATURE_SVE)
                    if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                    {
                        if (gap >= 8)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const uint64_t N = svcntd();
                            std::size_t j = 0;
                            for (; j + N <= gap; j += N)
                            {
                                svuint64_t vx = svld1_u64(svptrue_b64(), x + j);
                                svuint64_t vy = svld1_u64(svptrue_b64(), y + j);
                                svuint64_t sum = svadd_u64_x(svptrue_b64(), vx, vy);
                                svuint64_t xo = sve_guard(sum, two_p, svptrue_b64());
                                svuint64_t diff = svsub_u64_x(svptrue_b64(),
                                    svadd_u64_x(svptrue_b64(), vx, svdup_n_u64(two_p)), vy);
                                svuint64_t yo = sve_mulmod_lazy(diff, r.operand, r.quotient, p, svptrue_b64());
                                svst1_u64(svptrue_b64(), x + j, xo);
                                svst1_u64(svptrue_b64(), y + j, yo);
                            }
                            while (j < gap)
                            {
                                svbool_t pg = svwhilelt_b64(j, gap);
                                svuint64_t vx = svld1_u64(pg, x + j);
                                svuint64_t vy = svld1_u64(pg, y + j);
                                svuint64_t sum = svadd_u64_x(pg, vx, vy);
                                svuint64_t xo = sve_guard(sum, two_p, pg);
                                svuint64_t diff = svsub_u64_x(pg,
                                    svadd_u64_x(pg, vx, svdup_n_u64(two_p)), vy);
                                svuint64_t yo = sve_mulmod_lazy(diff, r.operand, r.quotient, p, pg);
                                svst1_u64(pg, x + j, xo);
                                svst1_u64(pg, y + j, yo);
                                j += N;
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4)
                            {
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                        }
                    }
                    else
#endif
                    {
                        if (gap < 4)
                        {
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                u = *x;
                                v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                        }
                        else
                        {
                            for (std::size_t j = 0; j < gap; j += 4) // gap is power of 2, gap%4==0
                            {
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                                u = *x; v = *y;
                                *x++ = arithmetic_.guard(arithmetic_.add(u, v));
                                *y++ = arithmetic_.mul_root(arithmetic_.sub(u, v), r);
                            }
                        }
                    }
                }
            }

        private:
            Arithmetic<ValueType, RootType, ScalarType> arithmetic_;
        };
    } // namespace util
} // namespace seal
