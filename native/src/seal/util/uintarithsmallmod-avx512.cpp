// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/uintarith.h"
#include "seal/util/uintarithmod.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <numeric>
#include <random>
#include <tuple>
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
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
                auto x1_alloc(allocate_uint(uint64_count - 2, pool));
                uint64_t *x1 = x1_alloc.get();
                uint64_t x2[2];
                auto quot_alloc(allocate_uint(uint64_count, pool));
                uint64_t *quot = quot_alloc.get();
                auto rem_alloc(allocate_uint(uint64_count, pool));
                uint64_t *rem = rem_alloc.get();
                set_uint(numerator + 2, uint64_count - 2, x1);
                set_uint(numerator, 2, x2);

                multiply_uint(x1, uint64_count - 2, &modulus.const_ratio()[0], 2, uint64_count, quot);
                multiply_uint(x1, uint64_count - 2, modulus.const_ratio()[2], uint64_count - 1, rem);
                add_uint(rem, uint64_count - 1, x2, 2, 0, uint64_count, rem);

                size_t remainder_uint64_count = get_significant_uint64_count_uint(rem, uint64_count);
                divide_uint_mod_inplace(rem, modulus, remainder_uint64_count, quotient, pool);
                add_uint(quotient, quot, uint64_count, quotient);
                *numerator = rem[0];

                return;
            }
        }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
        // AVX-512 has no vpmulhq for 64-bit integers.
        // Emulate high 64 bits of 64x64 multiply via 32-bit decomposition.
        // a*b = p_hh*2^64 + (p_hl+p_lh)*2^32 + p_ll
        // high64 = p_hh + mid_hi + (mid_carry << 32) + t_carry
        static inline __m512i avx512_mulhi_epu64(__m512i a, __m512i b)
        {
            __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFF);
            __m512i a_lo = _mm512_and_si512(a, mask32);
            __m512i a_hi = _mm512_srli_epi64(a, 32);
            __m512i b_lo = _mm512_and_si512(b, mask32);
            __m512i b_hi = _mm512_srli_epi64(b, 32);

            __m512i p_hh = _mm512_mul_epu32(a_hi, b_hi);
            __m512i p_hl = _mm512_mul_epu32(a_hi, b_lo);
            __m512i p_lh = _mm512_mul_epu32(a_lo, b_hi);
            __m512i p_ll = _mm512_mul_epu32(a_lo, b_lo);

            // mid = p_hl + p_lh (may overflow 64-bit)
            __m512i mid = _mm512_add_epi64(p_hl, p_lh);
            __mmask8 mid_carry = _mm512_cmplt_epu64_mask(mid, p_hl);

            __m512i mid_lo = _mm512_and_si512(mid, mask32);
            __m512i mid_hi = _mm512_srli_epi64(mid, 32);

            // t = p_ll + (mid_lo << 32)
            __m512i t = _mm512_add_epi64(p_ll, _mm512_slli_epi64(mid_lo, 32));
            __mmask8 t_carry = _mm512_cmplt_epu64_mask(t, p_ll);

            // high64 = p_hh + mid_hi + (mid_carry << 32) + t_carry
            __m512i hi = _mm512_add_epi64(p_hh, mid_hi);
            hi = _mm512_add_epi64(hi,
                _mm512_slli_epi64(_mm512_maskz_set1_epi64(mid_carry, 1), 32));
            hi = _mm512_add_epi64(hi,
                _mm512_maskz_set1_epi64(t_carry, 1));
            return hi;
        }

        // 128-bit add: (lo,hi) += (add_lo,add_hi) per lane
        static inline void avx512_add_u128(
            __m512i &lo, __m512i &hi, __m512i addend_lo, __m512i addend_hi)
        {
            __m512i sum_lo = _mm512_add_epi64(lo, addend_lo);
            __mmask8 carry = _mm512_cmplt_epu64_mask(sum_lo, lo);
            __m512i carry_val = _mm512_maskz_set1_epi64(carry, 1);
            hi = _mm512_add_epi64(_mm512_add_epi64(hi, addend_hi), carry_val);
            lo = sum_lo;
        }
#endif

        uint64_t dot_product_mod(
            const uint64_t *operand1, const uint64_t *operand2, size_t count, const Modulus &modulus)
        {
            static_assert(SEAL_MULTIPLY_ACCUMULATE_MOD_MAX >= 16, "SEAL_MULTIPLY_ACCUMULATE_MOD_MAX");

#if defined(__AVX512F__) && defined(__AVX512DQ__)
            if (count == 0)
                return 0;

            constexpr uint64_t N = 8; // AVX-512 = 8 x uint64

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

            // AVX-512 vectorized path for count > 16
            __m512i acc_lo = _mm512_setzero_si512();
            __m512i acc_hi = _mm512_setzero_si512();

            size_t i = 0;
            // 4x unrolled main loop
            while (i + 4 * N <= count)
            {
                __m512i v1_0 = _mm512_loadu_si512(operand1 + i);
                __m512i v2_0 = _mm512_loadu_si512(operand2 + i);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1_0, v2_0),
                    avx512_mulhi_epu64(v1_0, v2_0));

                __m512i v1_1 = _mm512_loadu_si512(operand1 + i + N);
                __m512i v2_1 = _mm512_loadu_si512(operand2 + i + N);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1_1, v2_1),
                    avx512_mulhi_epu64(v1_1, v2_1));

                __m512i v1_2 = _mm512_loadu_si512(operand1 + i + 2 * N);
                __m512i v2_2 = _mm512_loadu_si512(operand2 + i + 2 * N);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1_2, v2_2),
                    avx512_mulhi_epu64(v1_2, v2_2));

                __m512i v1_3 = _mm512_loadu_si512(operand1 + i + 3 * N);
                __m512i v2_3 = _mm512_loadu_si512(operand2 + i + 3 * N);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1_3, v2_3),
                    avx512_mulhi_epu64(v1_3, v2_3));

                i += 4 * N;
            }

            // Handle remaining full vectors
            while (i + N <= count)
            {
                __m512i v1 = _mm512_loadu_si512(operand1 + i);
                __m512i v2 = _mm512_loadu_si512(operand2 + i);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1, v2),
                    avx512_mulhi_epu64(v1, v2));
                i += N;
            }

            // Handle tail elements with masked load
            if (i < count)
            {
                __mmask8 mask = static_cast<__mmask8>((1ULL << (count - i)) - 1);
                __m512i v1 = _mm512_maskz_loadu_epi64(mask, operand1 + i);
                __m512i v2 = _mm512_maskz_loadu_epi64(mask, operand2 + i);
                avx512_add_u128(acc_lo, acc_hi,
                    _mm512_mullo_epi64(v1, v2),
                    avx512_mulhi_epu64(v1, v2));
            }

            // Horizontal reduction: vector -> scalar 128-bit
            uint64_t vec_lo[8];
            uint64_t vec_hi[8];
            _mm512_storeu_si512(vec_lo, acc_lo);
            _mm512_storeu_si512(vec_hi, acc_hi);

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
