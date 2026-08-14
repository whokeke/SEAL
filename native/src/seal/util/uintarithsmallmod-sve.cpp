// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/uintarith.h"
#include "seal/util/uintarithmod.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <numeric>
#include <random>
#include <tuple>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        uint64_t exponentiate_uint_mod(uint64_t operand, uint64_t exponent, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (operand >= modulus.value())
            {
                throw invalid_argument("operand");
            }
#endif
            // Fast cases
            if (exponent == 0)
            {
                // Result is supposed to be only one digit
                return 1;
            }

            if (exponent == 1)
            {
                return operand;
            }

            // Perform binary exponentiation.
            uint64_t power = operand;
            uint64_t product = 0;
            uint64_t intermediate = 1;

            // Initially: power = operand and intermediate = 1, product is irrelevant.
            while (true)
            {
                if (exponent & 1)
                {
                    product = multiply_uint_mod(power, intermediate, modulus);
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = multiply_uint_mod(power, power, modulus);
                swap(product, power);
            }
            return intermediate;
        }

        void divide_uint_mod_inplace(
            uint64_t *numerator, const Modulus &modulus, size_t uint64_count, uint64_t *quotient, MemoryPool &pool)
        {
            // Handle base cases
            if (uint64_count == 2)
            {
                divide_uint128_inplace(numerator, modulus.value(), quotient);
                return;
            }
            else if (uint64_count == 1)
            {
                *numerator = barrett_reduce_64(*numerator, modulus);
                *quotient = *numerator / modulus.value();
                return;
            }
            else
            {
                // If uint64_count > 2.
                // x = numerator = x1 * 2^128 + x2.
                // 2^128 = A*value + B.

                auto x1_alloc(allocate_uint(uint64_count - 2, pool));
                uint64_t *x1 = x1_alloc.get();
                uint64_t x2[2];
                auto quot_alloc(allocate_uint(uint64_count, pool));
                uint64_t *quot = quot_alloc.get();
                auto rem_alloc(allocate_uint(uint64_count, pool));
                uint64_t *rem = rem_alloc.get();
                set_uint(numerator + 2, uint64_count - 2, x1);
                set_uint(numerator, 2, x2); // x2 = (num) % 2^128.

                multiply_uint(x1, uint64_count - 2, &modulus.const_ratio()[0], 2, uint64_count, quot); // x1*A.
                multiply_uint(x1, uint64_count - 2, modulus.const_ratio()[2], uint64_count - 1, rem); // x1*B
                add_uint(rem, uint64_count - 1, x2, 2, 0, uint64_count, rem); // x1*B + x2;

                size_t remainder_uint64_count = get_significant_uint64_count_uint(rem, uint64_count);
                divide_uint_mod_inplace(rem, modulus, remainder_uint64_count, quotient, pool);
                add_uint(quotient, quot, uint64_count, quotient);
                *numerator = rem[0];

                return;
            }
        }

#ifdef __ARM_FEATURE_SVE
        static inline void sve_add_u128(
            svuint64_t &lo, svuint64_t &hi, svuint64_t addend_lo, svuint64_t addend_hi)
        {
            // 128-bit addition: (lo, hi) += (addend_lo, addend_hi)
            svuint64_t sum_lo = svadd_u64_x(svptrue_b64(), lo, addend_lo);
            svbool_t carry = svcmplt_u64(svptrue_b64(), sum_lo, lo);
            svuint64_t carry_val = svsel_u64(carry, svdup_n_u64(1), svdup_n_u64(0));
            hi = svadd_u64_x(svptrue_b64(), hi, addend_hi);
            hi = svadd_u64_x(svptrue_b64(), hi, carry_val);
            lo = sum_lo;
        }
#endif

        uint64_t dot_product_mod(
            const uint64_t *operand1, const uint64_t *operand2, size_t count, const Modulus &modulus)
        {
            static_assert(SEAL_MULTIPLY_ACCUMULATE_MOD_MAX >= 16, "SEAL_MULTIPLY_ACCUMULATE_MOD_MAX");

#ifdef __ARM_FEATURE_SVE
            if (count == 0)
                return 0;

            // For small counts, scalar template-unrolled code is faster
            if (count <= 16)
            {
                unsigned long long accumulator[2]{ 0, 0 };
                switch (count)
                {
                case 0:
                    return 0;
                case 1:
                    multiply_accumulate_uint64<1>(operand1, operand2, accumulator);
                    break;
                case 2:
                    multiply_accumulate_uint64<2>(operand1, operand2, accumulator);
                    break;
                case 3:
                    multiply_accumulate_uint64<3>(operand1, operand2, accumulator);
                    break;
                case 4:
                    multiply_accumulate_uint64<4>(operand1, operand2, accumulator);
                    break;
                case 5:
                    multiply_accumulate_uint64<5>(operand1, operand2, accumulator);
                    break;
                case 6:
                    multiply_accumulate_uint64<6>(operand1, operand2, accumulator);
                    break;
                case 7:
                    multiply_accumulate_uint64<7>(operand1, operand2, accumulator);
                    break;
                case 8:
                    multiply_accumulate_uint64<8>(operand1, operand2, accumulator);
                    break;
                case 9:
                    multiply_accumulate_uint64<9>(operand1, operand2, accumulator);
                    break;
                case 10:
                    multiply_accumulate_uint64<10>(operand1, operand2, accumulator);
                    break;
                case 11:
                    multiply_accumulate_uint64<11>(operand1, operand2, accumulator);
                    break;
                case 12:
                    multiply_accumulate_uint64<12>(operand1, operand2, accumulator);
                    break;
                case 13:
                    multiply_accumulate_uint64<13>(operand1, operand2, accumulator);
                    break;
                case 14:
                    multiply_accumulate_uint64<14>(operand1, operand2, accumulator);
                    break;
                case 15:
                    multiply_accumulate_uint64<15>(operand1, operand2, accumulator);
                    break;
                case 16:
                largest_case:
                    multiply_accumulate_uint64<16>(operand1, operand2, accumulator);
                    break;
                default:
                    accumulator[0] = dot_product_mod(operand1 + 16, operand2 + 16, count - 16, modulus);
                    goto largest_case;
                };
                return barrett_reduce_128(accumulator, modulus);
            }

            const uint64_t N = svcntd();

            // Single accumulator pair with 4x-unrolled loop body for Neoverse V2
            svuint64_t acc_lo = svdup_n_u64(0);
            svuint64_t acc_hi = svdup_n_u64(0);

            size_t i = 0;
            // 4x unrolled main loop
            while (i + 4 * N <= count)
            {
                svuint64_t v1_0 = svld1_u64(svptrue_b64(), operand1 + i);
                svuint64_t v2_0 = svld1_u64(svptrue_b64(), operand2 + i);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(svptrue_b64(), v1_0, v2_0), svmulh_u64_z(svptrue_b64(), v1_0, v2_0));

                svuint64_t v1_1 = svld1_u64(svptrue_b64(), operand1 + i + N);
                svuint64_t v2_1 = svld1_u64(svptrue_b64(), operand2 + i + N);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(svptrue_b64(), v1_1, v2_1), svmulh_u64_z(svptrue_b64(), v1_1, v2_1));

                svuint64_t v1_2 = svld1_u64(svptrue_b64(), operand1 + i + 2 * N);
                svuint64_t v2_2 = svld1_u64(svptrue_b64(), operand2 + i + 2 * N);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(svptrue_b64(), v1_2, v2_2), svmulh_u64_z(svptrue_b64(), v1_2, v2_2));

                svuint64_t v1_3 = svld1_u64(svptrue_b64(), operand1 + i + 3 * N);
                svuint64_t v2_3 = svld1_u64(svptrue_b64(), operand2 + i + 3 * N);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(svptrue_b64(), v1_3, v2_3), svmulh_u64_z(svptrue_b64(), v1_3, v2_3));

                i += 4 * N;
            }

            // Handle remaining full vectors
            while (i + N <= count)
            {
                svuint64_t v1 = svld1_u64(svptrue_b64(), operand1 + i);
                svuint64_t v2 = svld1_u64(svptrue_b64(), operand2 + i);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(svptrue_b64(), v1, v2), svmulh_u64_z(svptrue_b64(), v1, v2));
                i += N;
            }

            // Handle tail elements with predicate
            if (i < count)
            {
                svbool_t pg = svwhilelt_b64_u64(i, count);
                svuint64_t v1 = svld1_u64(pg, operand1 + i);
                svuint64_t v2 = svld1_u64(pg, operand2 + i);
                sve_add_u128(acc_lo, acc_hi, svmul_u64_z(pg, v1, v2), svmulh_u64_z(pg, v1, v2));
            }

            // Horizontal reduction: vector -> scalar 128-bit
            uint64_t vec_lo[8]; // max svcntd() for up to 512-bit SVE
            uint64_t vec_hi[8];
            svst1_u64(svptrue_b64(), vec_lo, acc_lo);
            svst1_u64(svptrue_b64(), vec_hi, acc_hi);

            unsigned long long accumulator[2] = {0, 0};
            for (uint64_t j = 0; j < N; j++)
            {
                unsigned long long qword[2] = {vec_lo[j], vec_hi[j]};
                add_uint128(qword, accumulator, accumulator);
            }

            return barrett_reduce_128(accumulator, modulus);
#else
            unsigned long long accumulator[2]{ 0, 0 };
            switch (count)
            {
            case 0:
                return 0;
            case 1:
                multiply_accumulate_uint64<1>(operand1, operand2, accumulator);
                break;
            case 2:
                multiply_accumulate_uint64<2>(operand1, operand2, accumulator);
                break;
            case 3:
                multiply_accumulate_uint64<3>(operand1, operand2, accumulator);
                break;
            case 4:
                multiply_accumulate_uint64<4>(operand1, operand2, accumulator);
                break;
            case 5:
                multiply_accumulate_uint64<5>(operand1, operand2, accumulator);
                break;
            case 6:
                multiply_accumulate_uint64<6>(operand1, operand2, accumulator);
                break;
            case 7:
                multiply_accumulate_uint64<7>(operand1, operand2, accumulator);
                break;
            case 8:
                multiply_accumulate_uint64<8>(operand1, operand2, accumulator);
                break;
            case 9:
                multiply_accumulate_uint64<9>(operand1, operand2, accumulator);
                break;
            case 10:
                multiply_accumulate_uint64<10>(operand1, operand2, accumulator);
                break;
            case 11:
                multiply_accumulate_uint64<11>(operand1, operand2, accumulator);
                break;
            case 12:
                multiply_accumulate_uint64<12>(operand1, operand2, accumulator);
                break;
            case 13:
                multiply_accumulate_uint64<13>(operand1, operand2, accumulator);
                break;
            case 14:
                multiply_accumulate_uint64<14>(operand1, operand2, accumulator);
                break;
            case 15:
                multiply_accumulate_uint64<15>(operand1, operand2, accumulator);
                break;
            case 16:
            largest_case:
                multiply_accumulate_uint64<16>(operand1, operand2, accumulator);
                break;
            default:
                accumulator[0] = dot_product_mod(operand1 + 16, operand2 + 16, count - 16, modulus);
                goto largest_case;
            };
            return barrett_reduce_128(accumulator, modulus);
#endif
        }
    } // namespace util
} // namespace seal
