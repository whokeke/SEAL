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

#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

namespace seal
{
    namespace util
    {
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

        template <typename ValueType, typename RootType, typename ScalarType>
        class DWTHandler
        {
        public:
            DWTHandler() {}

            DWTHandler(const Arithmetic<ValueType, RootType, ScalarType> &num_struct) : arithmetic_(num_struct) {}

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
                    else
                    {
                        auto scalar_4way = [&]() {
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
                        };
#if defined(__ARM_FEATURE_SVE2)
                        if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const svuint64_t p_sv = svdup_n_u64(p);
                            const svuint64_t tp_sv = svdup_n_u64(two_p);
                            const std::size_t vl = svcntd();
                            if (gap >= vl)
                            {
                                for (std::size_t i = 0; i < m; i++)
                                {
                                    r = *++roots;
                                    x = values + offset;
                                    y = x + gap;
                                    const svuint64_t rop_sv = svdup_n_u64(r.operand);
                                    const svuint64_t rq_sv = svdup_n_u64(r.quotient);
                                    std::size_t j = 0;
                                    for (; j + vl <= gap; j += vl)
                                    {
                                        svbool_t pg = svptrue_b64();
                                        svuint64_t vx = svld1_u64(pg, x + j);
                                        svuint64_t vy = svld1_u64(pg, y + j);
                                        svbool_t ge = svcmpge_u64(pg, vx, tp_sv);
                                        svuint64_t gx = svsel_u64(ge, svsub_u64_x(pg, vx, tp_sv), vx);
                                        svuint64_t t1 = svmulh_u64_x(pg, vy, rq_sv);
                                        svuint64_t lo = svmul_u64_x(pg, vy, rop_sv);
                                        svuint64_t hi = svmul_u64_x(pg, t1, p_sv);
                                        svuint64_t vv = svsub_u64_x(pg, lo, hi);
                                        svst1_u64(pg, x + j, svadd_u64_x(pg, gx, vv));
                                        svst1_u64(pg, y + j, svsub_u64_x(pg, svadd_n_u64_x(pg, gx, two_p), vv));
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
                                scalar_4way();
                            }
                        }
                        else
                        {
                            scalar_4way();
                        }
#else
                        scalar_4way();
#endif
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
                    else
                    {
                        auto scalar_4way = [&]() {
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
                        };
#if defined(__ARM_FEATURE_SVE2)
                        if constexpr (std::is_same_v<ValueType, std::uint64_t>)
                        {
                            const std::uint64_t p = arithmetic_.modulus_.value();
                            const std::uint64_t two_p = arithmetic_.two_times_modulus_;
                            const svuint64_t p_sv = svdup_n_u64(p);
                            const svuint64_t tp_sv = svdup_n_u64(two_p);
                            const std::size_t vl = svcntd();
                            if (gap >= vl)
                            {
                                for (std::size_t i = 0; i < m; i++)
                                {
                                    r = *++roots;
                                    x = values + offset;
                                    y = x + gap;
                                    const svuint64_t rop_sv = svdup_n_u64(r.operand);
                                    const svuint64_t rq_sv = svdup_n_u64(r.quotient);
                                    std::size_t j = 0;
                                    for (; j + vl <= gap; j += vl)
                                    {
                                        svbool_t pg = svptrue_b64();
                                        svuint64_t vx = svld1_u64(pg, x + j);
                                        svuint64_t vy = svld1_u64(pg, y + j);
                                        svuint64_t sum = svadd_u64_x(pg, vx, vy);
                                        svbool_t ge = svcmpge_u64(pg, sum, tp_sv);
                                        svuint64_t xo = svsel_u64(ge, svsub_u64_x(pg, sum, tp_sv), sum);
                                        svuint64_t diff = svsub_u64_x(pg, svadd_n_u64_x(pg, vx, two_p), vy);
                                        svuint64_t t1 = svmulh_u64_x(pg, diff, rq_sv);
                                        svuint64_t lo = svmul_u64_x(pg, diff, rop_sv);
                                        svuint64_t hi = svmul_u64_x(pg, t1, p_sv);
                                        svuint64_t yo = svsub_u64_x(pg, lo, hi);
                                        svst1_u64(pg, x + j, xo);
                                        svst1_u64(pg, y + j, yo);
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
                                scalar_4way();
                            }
                        }
                        else
                        {
                            scalar_4way();
                        }
#else
                        scalar_4way();
#endif
                    }
                    gap <<= 1;
                }

                if (scalar != nullptr)
                {
                    r = *++roots;
                    RootType scaled_r = arithmetic_.mul_root_scalar(r, *scalar);
                    x = values;
                    y = x + gap;
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
                {
                    r = *++roots;
                    x = values;
                    y = x + gap;
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
            }

        private:
            Arithmetic<ValueType, RootType, ScalarType> arithmetic_;
        };
    } // namespace util
} // namespace seal
