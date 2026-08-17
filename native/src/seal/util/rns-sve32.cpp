// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/common.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rns.h"
#include "seal/util/uintarithmod.h"
#include "seal/util/uintarithsmallmod.h"
#include <algorithm>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
namespace {
        // SVE32 schoolbook: 64x64->high64 via 32-bit decomposition
        // Uses SVE2 UMULLB/UMULLT for 32x32->64 widening multiply (inline asm)
        static inline svuint64_t sve32_mulhi_u64(svbool_t pg, svuint64_t a, svuint64_t b)
        {
            svuint32_t a32 = svreinterpret_u32_u64(a);
            svuint32_t b32 = svreinterpret_u32_u64(b);
            svuint64_t p_ll, p_hh;
            __asm__ __volatile__("umullb %0.d, %1.s, %2.s" : "=w"(p_ll) : "w"(a32), "w"(b32));
            __asm__ __volatile__("umullt %0.d, %1.s, %2.s" : "=w"(p_hh) : "w"(a32), "w"(b32));

            svuint64_t a_hi = svlsr_n_u64_x(pg, a, 32);
            svuint64_t b_hi = svlsr_n_u64_x(pg, b, 32);
            svuint32_t a_hi32 = svreinterpret_u32_u64(a_hi);
            svuint32_t b_hi32 = svreinterpret_u32_u64(b_hi);
            svuint64_t p_hl, p_lh;
            __asm__ __volatile__("umullb %0.d, %1.s, %2.s" : "=w"(p_hl) : "w"(a_hi32), "w"(b32));
            __asm__ __volatile__("umullb %0.d, %1.s, %2.s" : "=w"(p_lh) : "w"(a32), "w"(b_hi32));

            svuint64_t mask32 = svdup_n_u64(0xFFFFFFFF);
            svuint64_t mid = svadd_u64_x(pg, p_hl, p_lh);
            svbool_t mid_carry = svcmplt_u64(pg, mid, p_hl);
            svuint64_t mid_lo = svand_u64_x(pg, mid, mask32);
            svuint64_t mid_hi = svlsr_n_u64_x(pg, mid, 32);
            svuint64_t t = svadd_u64_x(pg, p_ll, svlsl_n_u64_x(pg, mid_lo, 32));
            svbool_t t_carry = svcmplt_u64(pg, t, p_ll);
            svuint64_t hi = svadd_u64_x(pg, p_hh, mid_hi);
            svuint64_t mc = svsel_u64(mid_carry, svdup_n_u64(1), svdup_n_u64(0));
            hi = svadd_u64_x(pg, hi, svlsl_n_u64_x(pg, mc, 32));
            svuint64_t tc = svsel_u64(t_carry, svdup_n_u64(1), svdup_n_u64(0));
            hi = svadd_u64_x(pg, hi, tc);
            return hi;
        }
    // SVE helper: 128-bit addition (lo, hi) += (addend_lo, addend_hi) per-lane
    static inline void sve_add_u128(
        svuint64_t &lo, svuint64_t &hi, svuint64_t addend_lo, svuint64_t addend_hi, svbool_t pg)
    {
        svuint64_t sum_lo = svadd_u64_x(pg, lo, addend_lo);
        svbool_t carry = svcmplt_u64(pg, sum_lo, lo);
        svuint64_t carry_val = svsel_u64(carry, svdup_n_u64(1), svdup_n_u64(0));
        hi = svadd_u64_x(pg, hi, addend_hi);
        hi = svadd_u64_x(pg, hi, carry_val);
        lo = sum_lo;
    }

    // SVE helper: 128-bit Barrett reduction of (acc_lo, acc_hi) mod modulus
    static inline svuint64_t sve_barrett_reduce_128(
        svuint64_t acc_lo, svuint64_t acc_hi,
        svuint64_t v_mod, svuint64_t v_ratio0, svuint64_t v_ratio1, svbool_t pg)
    {
        svuint64_t r0_hi = sve32_mulhi_u64(pg, acc_lo, v_ratio0);
        svuint64_t r1_lo = svmul_u64_x(pg, acc_lo, v_ratio1);
        svuint64_t r1_hi = sve32_mulhi_u64(pg, acc_lo, v_ratio1);

        svuint64_t tmp1 = svadd_u64_x(pg, r1_lo, r0_hi);
        svbool_t c1 = svcmplt_u64(pg, tmp1, r1_lo);
        svuint64_t c1v = svsel_u64(c1, svdup_n_u64(1), svdup_n_u64(0));
        svuint64_t tmp3 = svadd_u64_x(pg, r1_hi, c1v);

        svuint64_t r0p_lo = svmul_u64_x(pg, acc_hi, v_ratio0);
        svuint64_t r0p_hi = sve32_mulhi_u64(pg, acc_hi, v_ratio0);

        svuint64_t tmp1_new = svadd_u64_x(pg, tmp1, r0p_lo);
        svbool_t c2 = svcmplt_u64(pg, tmp1_new, tmp1);
        svuint64_t c2v = svsel_u64(c2, svdup_n_u64(1), svdup_n_u64(0));
        svuint64_t carry_val = svadd_u64_x(pg, r0p_hi, c2v);

        svuint64_t q_est = svmul_u64_x(pg, acc_hi, v_ratio1);
        q_est = svadd_u64_x(pg, q_est, tmp3);
        q_est = svadd_u64_x(pg, q_est, carry_val);

        svuint64_t prod = svmul_u64_x(pg, q_est, v_mod);
        svuint64_t res = svsub_u64_x(pg, acc_lo, prod);

        svbool_t ge = svcmpge_u64(pg, res, v_mod);
        return svsub_u64_m(ge, res, v_mod);
    }
} // anonymous namespace
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        RNSBase::RNSBase(const vector<Modulus> &rnsbase, MemoryPoolHandle pool)
            : pool_(std::move(pool)), size_(rnsbase.size())
        {
            if (!size_)
            {
                throw invalid_argument("rnsbase cannot be empty");
            }
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }

            for (size_t i = 0; i < rnsbase.size(); i++)
            {
                // The base elements cannot be zero
                if (rnsbase[i].is_zero())
                {
                    throw invalid_argument("rnsbase is invalid");
                }

                for (size_t j = 0; j < i; j++)
                {
                    // The base must be coprime
                    if (!are_coprime(rnsbase[i].value(), rnsbase[j].value()))
                    {
                        throw invalid_argument("rnsbase is invalid");
                    }
                }
            }

            // Base is good; now copy it over to rnsbase_
            base_ = allocate<Modulus>(size_, pool_);
            copy_n(rnsbase.cbegin(), size_, base_.get());

            // Initialize CRT data
            if (!initialize())
            {
                throw invalid_argument("rnsbase is invalid");
            }
        }

        RNSBase::RNSBase(const RNSBase &copy, MemoryPoolHandle pool) : pool_(std::move(pool)), size_(copy.size_)
        {
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }

            // Copy over the base
            base_ = allocate<Modulus>(size_, pool_);
            copy_n(copy.base_.get(), size_, base_.get());

            // Copy over CRT data
            base_prod_ = allocate_uint(size_, pool_);
            set_uint(copy.base_prod_.get(), size_, base_prod_.get());

            punctured_prod_array_ = allocate_uint(size_ * size_, pool_);
            set_uint(copy.punctured_prod_array_.get(), size_ * size_, punctured_prod_array_.get());

            inv_punctured_prod_mod_base_array_ = allocate<MultiplyUIntModOperand>(size_, pool_);
            copy_n(copy.inv_punctured_prod_mod_base_array_.get(), size_, inv_punctured_prod_mod_base_array_.get());
        }

        bool RNSBase::contains(const Modulus &value) const noexcept
        {
            bool result = false;
            SEAL_ITERATE(iter(base_), size_, [&](auto &I) { result = result || (I == value); });
            return result;
        }

        bool RNSBase::is_subbase_of(const RNSBase &superbase) const noexcept
        {
            bool result = true;
            SEAL_ITERATE(iter(base_), size_, [&](auto &I) { result = result && superbase.contains(I); });
            return result;
        }

        RNSBase RNSBase::extend(const Modulus &value) const
        {
            if (value.is_zero())
            {
                throw invalid_argument("value cannot be zero");
            }

            SEAL_ITERATE(iter(base_), size_, [&](auto I) {
                // The base must be coprime
                if (!are_coprime(I.value(), value.value()))
                {
                    throw logic_error("cannot extend by given value");
                }
            });

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = add_safe(size_, size_t(1));
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_, newbase.base_.get());

            // Extend with value
            newbase.base_[newbase.size_ - 1] = value;

            // Initialize CRT data
            if (!newbase.initialize())
            {
                throw logic_error("cannot extend by given value");
            }

            return newbase;
        }

        RNSBase RNSBase::extend(const RNSBase &other) const
        {
            // The bases must be coprime
            for (size_t i = 0; i < other.size_; i++)
            {
                for (size_t j = 0; j < size_; j++)
                {
                    if (!are_coprime(other[i].value(), base_[j].value()))
                    {
                        throw invalid_argument("rnsbase is invalid");
                    }
                }
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = add_safe(size_, other.size_);
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_, newbase.base_.get());

            // Extend with other base
            copy_n(other.base_.get(), other.size_, newbase.base_.get() + size_);

            // Initialize CRT data
            if (!newbase.initialize())
            {
                throw logic_error("cannot extend by given base");
            }

            return newbase;
        }

        RNSBase RNSBase::drop() const
        {
            if (size_ == 1)
            {
                throw logic_error("cannot drop from base of size 1");
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = size_ - 1;
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_ - 1, newbase.base_.get());

            // Initialize CRT data
            newbase.initialize();

            return newbase;
        }

        RNSBase RNSBase::drop(const Modulus &value) const
        {
            if (size_ == 1)
            {
                throw logic_error("cannot drop from base of size 1");
            }
            if (!contains(value))
            {
                throw logic_error("base does not contain value");
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = size_ - 1;
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            size_t source_index = 0;
            size_t dest_index = 0;
            while (dest_index < size_ - 1)
            {
                if (base_[source_index] != value)
                {
                    newbase.base_[dest_index] = base_[source_index];
                    dest_index++;
                }
                source_index++;
            }

            // Initialize CRT data
            newbase.initialize();

            return newbase;
        }

        bool RNSBase::initialize()
        {
            // Verify that the size is not too large
            if (!product_fits_in(size_, size_))
            {
                return false;
            }

            base_prod_ = allocate_uint(size_, pool_);
            punctured_prod_array_ = allocate_zero_uint(size_ * size_, pool_);
            inv_punctured_prod_mod_base_array_ = allocate<MultiplyUIntModOperand>(size_, pool_);

            if (size_ > 1)
            {
                auto rnsbase_values = allocate<uint64_t>(size_, pool_);
                SEAL_ITERATE(iter(base_, rnsbase_values), size_, [&](auto I) { get<1>(I) = get<0>(I).value(); });

                // Create punctured products
                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);
                SEAL_ITERATE(iter(punctured_prod, size_t(0)), size_, [&](auto I) {
                    multiply_many_uint64_except(rnsbase_values.get(), size_, get<1>(I), get<0>(I).ptr(), pool_);
                });

                // Compute the full product
                auto temp_mpi(allocate_uint(size_, pool_));
                multiply_uint(punctured_prod_array_.get(), size_, base_[0].value(), size_, temp_mpi.get());
                set_uint(temp_mpi.get(), size_, base_prod_.get());

                // Compute inverses of punctured products mod primes
                bool invertible = true;
                SEAL_ITERATE(iter(punctured_prod, base_, inv_punctured_prod_mod_base_array_), size_, [&](auto I) {
                    uint64_t temp = modulo_uint(get<0>(I), size_, get<1>(I));
                    invertible = invertible && try_invert_uint_mod(temp, get<1>(I), temp);
                    get<2>(I).set(temp, get<1>(I));
                });

                return invertible;
            }

            // Case of a single prime
            base_prod_[0] = base_[0].value();
            punctured_prod_array_[0] = 1;
            inv_punctured_prod_mod_base_array_[0].set(1, base_[0]);

            return true;
        }

        void RNSBase::decompose(uint64_t *value, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                // Copy the value
                auto value_copy(allocate_uint(size_, pool));
                set_uint(value, size_, value_copy.get());

                SEAL_ITERATE(iter(value, base_), size_, [&](auto I) {
                    get<0>(I) = modulo_uint(value_copy.get(), size_, get<1>(I));
                });
            }
        }

        void RNSBase::decompose_array(uint64_t *value, size_t count, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                // Decompose an array of multi-precision integers into an array of arrays, one per each base element

                // Copy the input array into a temporary location and set a StrideIter pointing to it
                // Note that the stride size is size_
                SEAL_ALLOCATE_GET_STRIDE_ITER(value_copy, uint64_t, count, size_, pool);
                set_uint(value, count * size_, value_copy);

                // Note how value_copy and value_out have size_ and count reversed
                RNSIter value_out(value, count);

                // For each output RNS array (one per base element) ...
                SEAL_ITERATE(iter(base_, value_out), size_, [&](auto I) {
                    // For each multi-precision integer in value_copy ...
                    SEAL_ITERATE(iter(get<1>(I), value_copy), count, [&](auto J) {
                        // Reduce the multi-precision integer modulo the base element and write to value_out
                        get<0>(J) = modulo_uint(get<1>(J), size_, get<0>(I));
                    });
                });
            }
        }

        void RNSBase::compose(uint64_t *value, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                // Copy the value
                auto temp_value(allocate_uint(size_, pool));
                set_uint(value, size_, temp_value.get());

                // Clear the result
                set_zero_uint(size_, value);

                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);

                // Compose an array of integers (one per base element) into a single multi-precision integer
                auto temp_mpi(allocate_uint(size_, pool));
                SEAL_ITERATE(
                    iter(temp_value, inv_punctured_prod_mod_base_array_, punctured_prod, base_), size_, [&](auto I) {
                        uint64_t temp_prod = multiply_uint_mod(get<0>(I), get<1>(I), get<3>(I));
                        multiply_uint(get<2>(I), size_, temp_prod, size_, temp_mpi.get());
                        add_uint_uint_mod(temp_mpi.get(), value, base_prod_.get(), size_, value);
                    });
            }
        }

        void RNSBase::compose_array(uint64_t *value, size_t count, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                // Merge the coefficients first
                auto temp_array(allocate_uint(count * size_, pool));
                for (size_t i = 0; i < count; i++)
                {
                    for (size_t j = 0; j < size_; j++)
                    {
                        temp_array[j + (i * size_)] = value[(j * count) + i];
                    }
                }

                // Clear the result
                set_zero_uint(count * size_, value);

                StrideIter<uint64_t *> temp_array_iter(temp_array.get(), size_);
                StrideIter<uint64_t *> value_iter(value, size_);
                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);

                // Compose an array of RNS integers into a single array of multi-precision integers
                auto temp_mpi(allocate_uint(size_, pool));
                SEAL_ITERATE(iter(temp_array_iter, value_iter), count, [&](auto I) {
                    SEAL_ITERATE(
                        iter(get<0>(I), inv_punctured_prod_mod_base_array_, punctured_prod, base_), size_, [&](auto J) {
                            uint64_t temp_prod = multiply_uint_mod(get<0>(J), get<1>(J), get<3>(J));
                            multiply_uint(get<2>(J), size_, temp_prod, size_, temp_mpi.get());
                            add_uint_uint_mod(temp_mpi.get(), get<1>(I), base_prod_.get(), size_, get<1>(I));
                        });
                });
            }
        }

        void BaseConverter::fast_convert(ConstCoeffIter in, CoeffIter out, MemoryPoolHandle pool) const
        {
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, ibase_size, pool);
            SEAL_ITERATE(
                iter(temp, in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base()), ibase_size,
                [&](auto I) { get<0>(I) = multiply_uint_mod(get<1>(I), get<2>(I), get<3>(I)); });

            // for (size_t j = 0; j < obase_size; j++)
            SEAL_ITERATE(iter(out, base_change_matrix_, obase_.base()), obase_size, [&](auto I) {
                get<0>(I) = dot_product_mod(temp, get<1>(I).get(), ibase_size, get<2>(I));
            });
        }

        void BaseConverter::fast_convert_array(ConstRNSIter in, RNSIter out, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (in.poly_modulus_degree() != out.poly_modulus_degree())
            {
                throw invalid_argument("in and out are incompatible");
            }
#endif
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();
            size_t count = in.poly_modulus_degree();

            // Note that the stride size is ibase_size
            SEAL_ALLOCATE_GET_STRIDE_ITER(temp, uint64_t, count, ibase_size, pool);

            SEAL_ITERATE(
                iter(in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base(), size_t(0)), ibase_size,
                [&](auto I) {
                    // The current ibase index
                    size_t ibase_index = get<3>(I);

                    if (get<1>(I).operand == 1)
                    {
                        // No multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                            // Reduce modulo ibase element
                            get<1>(J)[ibase_index] = barrett_reduce_64(get<0>(J), get<2>(I));
                        });
                    }
                    else
                    {
                        // Multiplication needed
#ifdef __ARM_FEATURE_SVE
                        {
                            const uint64_t *in_ptr = get<0>(I).ptr();
                            uint64_t *temp_base = temp.ptr();
                            const uint64_t p = get<2>(I).value();
                            const uint64_t N = svcntd();

                            svuint64_t sv_mod = svdup_n_u64(p);
                            svuint64_t sv_inv_op = svdup_n_u64(get<1>(I).operand);
                            svuint64_t sv_inv_q = svdup_n_u64(get<1>(I).quotient);

                            // temp[j][ibase_index] = (in[ibase_index][j] * inv_punctured_prod) mod ibase_mod
                            size_t j = 0;
                            while (j + N <= count)
                            {
                                // Contiguous read of in[ibase_index][j..j+N-1]
                                svuint64_t v_in = svld1_u64(svptrue_b64(), in_ptr + j);
                                // lazy Barrett mulmod: in * inv_op - p * hi(in, inv_q)
                                svuint64_t v_hi = sve32_mulhi_u64(svptrue_b64(), v_in, sv_inv_q);
                                svuint64_t v_lo = svmul_u64_x(svptrue_b64(), v_in, sv_inv_op);
                                v_lo = svmls_u64_x(svptrue_b64(), v_lo, sv_mod, v_hi);
                                // conditional subtract -> [0, p)
                                svbool_t ge = svcmpge_u64(svptrue_b64(), v_lo, sv_mod);
                                v_lo = svsub_u64_m(ge, v_lo, sv_mod);
                                // Strided scatter store: temp[(j+k)*ibase_size + ibase_index]
                                svuint64_t v_idx = svindex_u64(j * ibase_size + ibase_index, ibase_size);
                                svst1_scatter_u64index_u64(svptrue_b64(), temp_base, v_idx, v_lo);
                                j += N;
                            }
                            // Tail
                            if (j < count)
                            {
                                svbool_t pg = svwhilelt_b64(j, count);
                                svuint64_t v_in = svld1_u64(pg, in_ptr + j);
                                svuint64_t v_hi = sve32_mulhi_u64(pg, v_in, sv_inv_q);
                                svuint64_t v_lo = svmul_u64_x(pg, v_in, sv_inv_op);
                                v_lo = svmls_u64_x(pg, v_lo, sv_mod, v_hi);
                                svbool_t ge = svcmpge_u64(pg, v_lo, sv_mod);
                                v_lo = svsub_u64_m(ge, v_lo, sv_mod);
                                svuint64_t v_idx = svindex_u64(j * ibase_size + ibase_index, ibase_size);
                                svst1_scatter_u64index_u64(pg, temp_base, v_idx, v_lo);
                            }
                        }
#else
                        SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                            // Multiply coefficient of in with ibase_.inv_punctured_prod_mod_base_array_ element
                            get<1>(J)[ibase_index] = multiply_uint_mod(get<0>(J), get<1>(I), get<2>(I));
                        });
#endif
                    }
                });

#ifdef __ARM_FEATURE_SVE
            // SVE vectorized Phase 2: break the adds/adc carry-chain by using
            // sve_add_u128 for 128-bit accumulation without carry flags.
            // For each output component k, iterate over ibase elements i,
            // accumulating across j (coefficients) with SVE vectors.
            // This eliminates the serial carry dependency in the inner loop.
            for (size_t k = 0; k < obase_size; k++)
            {
                const uint64_t *base_change_ptr = base_change_matrix_[k].get();
                const Modulus &mod = obase_.base()[k];
                uint64_t *out_ptr = out[k].ptr();
                const uint64_t *temp_base = temp.ptr();
                const uint64_t N = svcntd();

                // SVE constants for Barrett reduction
                const svuint64_t sv_mod = svdup_n_u64(mod.value());
                const svuint64_t sv_ratio0 = svdup_n_u64(mod.const_ratio()[0]);
                const svuint64_t sv_ratio1 = svdup_n_u64(mod.const_ratio()[1]);

                size_t j = 0;
                // Process N coefficients at a time
                while (j + N <= count)
                {
                    svuint64_t acc_lo = svdup_n_u64(0);
                    svuint64_t acc_hi = svdup_n_u64(0);

                    // Inner loop: iterate over ibase elements (ibase_size is small)
                    for (size_t i = 0; i < ibase_size; i++)
                    {
                        // Broadcast base_change[k][i] to all lanes
                        svuint64_t bc_i = svdup_n_u64(base_change_ptr[i]);

                        // Gather-load temp[(j+0)*ibase_size+i], temp[(j+1)*ibase_size+i], ...
                        // stride = ibase_size across j values
                        svuint64_t v_indices = svindex_u64((j + 0) * ibase_size + i, ibase_size);
                        svuint64_t v_temp = svld1_gather_u64index_u64(svptrue_b64(), temp_base, v_indices);

                        // 128-bit multiply-accumulate (no carry flag dependency!)
                        svuint64_t prod_lo = svmul_u64_x(svptrue_b64(), v_temp, bc_i);
                        svuint64_t prod_hi = sve32_mulhi_u64(svptrue_b64(), v_temp, bc_i);
                        sve_add_u128(acc_lo, acc_hi, prod_lo, prod_hi, svptrue_b64());
                    }

                    // SVE Barrett reduce 128-bit (acc_lo, acc_hi) per lane
                    svuint64_t res = sve_barrett_reduce_128(acc_lo, acc_hi, sv_mod, sv_ratio0, sv_ratio1, svptrue_b64());

                    svst1_u64(svptrue_b64(), out_ptr + j, res);
                    j += N;
                }

                // Handle tail coefficients
                if (j < count)
                {
                    svbool_t pg = svwhilelt_b64(j, count);
                    svuint64_t acc_lo = svdup_n_u64(0);
                    svuint64_t acc_hi = svdup_n_u64(0);

                    for (size_t i = 0; i < ibase_size; i++)
                    {
                        svuint64_t bc_i = svdup_n_u64(base_change_ptr[i]);
                        svuint64_t v_indices = svindex_u64(j * ibase_size + i, ibase_size);
                        svuint64_t v_temp = svld1_gather_u64index_u64(pg, temp_base, v_indices);
                        svuint64_t prod_lo = svmul_u64_x(pg, v_temp, bc_i);
                        svuint64_t prod_hi = sve32_mulhi_u64(pg, v_temp, bc_i);
                        sve_add_u128(acc_lo, acc_hi, prod_lo, prod_hi, pg);
                    }

                    // Barrett reduce
                    svuint64_t res = sve_barrett_reduce_128(acc_lo, acc_hi, sv_mod, sv_ratio0, sv_ratio1, pg);
                    svst1_u64(pg, out_ptr + j, res);
                }
            }
#else
            SEAL_ITERATE(iter(out, base_change_matrix_, obase_.base()), obase_size, [&](auto I) {
                SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                    // Compute the base conversion sum modulo obase element
                    get<0>(J) = dot_product_mod(get<1>(J), get<1>(I).get(), ibase_size, get<2>(I));
                });
            });
#endif
        }

        // See "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme" (CT-RSA 2019) for details
        void BaseConverter::exact_convert_array(ConstRNSIter in, CoeffIter out, MemoryPoolHandle pool) const
        {
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();
            size_t count = in.poly_modulus_degree();

            if (obase_size != 1)
            {
                throw invalid_argument("out base in exact_convert_array must be one.");
            }

            // Note that the stride size is ibase_size
            SEAL_ALLOCATE_GET_STRIDE_ITER(temp, uint64_t, count, ibase_size, pool);

            // The iterator storing v
            SEAL_ALLOCATE_GET_STRIDE_ITER(v, double_t, count, ibase_size, pool);

            // Aggregated rounded v
            SEAL_ALLOCATE_GET_PTR_ITER(aggregated_rounded_v, uint64_t, count, pool);

            // Calculate [x_{i} * \hat{q_{i}}]_{q_{i}}
            SEAL_ITERATE(
                iter(in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base(), size_t(0)), ibase_size,
                [&](auto I) {
                    // The current ibase index
                    size_t ibase_index = get<3>(I);
                    double_t divisor = static_cast<double_t>(get<2>(I).value());

                    if (get<1>(I).operand == 1)
                    {
                        // No multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp, v), count, [&](auto J) {
                            // Reduce modulo ibase element
                            get<1>(J)[ibase_index] = barrett_reduce_64(get<0>(J), get<2>(I));
                            double_t dividend = static_cast<double_t>(get<1>(J)[ibase_index]);
                            get<2>(J)[ibase_index] = dividend / divisor;
                        });
                    }
                    else
                    {
                        // Multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp, v), count, [&](auto J) {
                            // Multiply coefficient of in with ibase_.inv_punctured_prod_mod_base_array_ element
                            get<1>(J)[ibase_index] = multiply_uint_mod(get<0>(J), get<1>(I), get<2>(I));
                            double_t dividend = static_cast<double_t>(get<1>(J)[ibase_index]);
                            get<2>(J)[ibase_index] = dividend / divisor;
                        });
                    }
                });

            // Aggrate v and rounding
            SEAL_ITERATE(iter(v, aggregated_rounded_v), count, [&](auto I) {
                // Otherwise a memory space of the last execution will be used.
                double_t aggregated_v = 0.0;
                for (size_t i = 0; i < ibase_size; ++i)
                {
                    aggregated_v += get<0>(I)[i];
                }
                aggregated_v += 0.5;
                get<1>(I) = static_cast<uint64_t>(aggregated_v);
            });

            auto p = obase_.base()[0];
            auto q_mod_p = modulo_uint(ibase_.base_prod(), ibase_size, p);
            auto base_change_matrix_first = base_change_matrix_[0].get();
            // Final multiplication
            SEAL_ITERATE(iter(out, temp, aggregated_rounded_v), count, [&](auto J) {
                // Compute the base conversion sum modulo obase element
                auto sum_mod_obase = dot_product_mod(get<1>(J), base_change_matrix_first, ibase_size, p);
                // Minus v*[q]_{p} mod p
                auto v_q_mod_p = multiply_uint_mod(get<2>(J), q_mod_p, p);
                get<0>(J) = sub_uint_mod(sum_mod_obase, v_q_mod_p, p);
            });
        }

        void BaseConverter::initialize()
        {
            // Verify that the size is not too large
            if (!product_fits_in(ibase_.size(), obase_.size()))
            {
                throw logic_error("invalid parameters");
            }

            // Create the base-change matrix rows
            base_change_matrix_ = allocate<Pointer<uint64_t>>(obase_.size(), pool_);

            SEAL_ITERATE(iter(base_change_matrix_, obase_.base()), obase_.size(), [&](auto I) {
                // Create the base-change matrix columns
                get<0>(I) = allocate_uint(ibase_.size(), pool_);

                StrideIter<const uint64_t *> ibase_punctured_prod_array(ibase_.punctured_prod_array(), ibase_.size());
                SEAL_ITERATE(iter(get<0>(I), ibase_punctured_prod_array), ibase_.size(), [&](auto J) {
                    // Base-change matrix contains the punctured products of ibase elements modulo the obase
                    get<0>(J) = modulo_uint(get<1>(J), ibase_.size(), get<1>(I));
                });
            });
        }

        RNSTool::RNSTool(
            size_t poly_modulus_degree, const RNSBase &coeff_modulus, const Modulus &plain_modulus,
            MemoryPoolHandle pool)
            : pool_(std::move(pool))
        {
#ifdef SEAL_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(poly_modulus_degree, coeff_modulus, plain_modulus);
        }

        void RNSTool::initialize(size_t poly_modulus_degree, const RNSBase &q, const Modulus &t)
        {
            // Return if q is out of bounds
            if (q.size() < SEAL_COEFF_MOD_COUNT_MIN || q.size() > SEAL_COEFF_MOD_COUNT_MAX)
            {
                throw invalid_argument("rnsbase is invalid");
            }

            // Return if coeff_count is not a power of two or out of bounds
            int coeff_count_power = get_power_of_two(poly_modulus_degree);
            if (coeff_count_power < 0 || poly_modulus_degree > SEAL_POLY_MOD_DEGREE_MAX ||
                poly_modulus_degree < SEAL_POLY_MOD_DEGREE_MIN)
            {
                throw invalid_argument("poly_modulus_degree is invalid");
            }

            t_ = t;
            coeff_count_ = poly_modulus_degree;

            // Allocate memory for the bases q, B, Bsk, Bsk U m_tilde, t_gamma
            size_t base_q_size = q.size();

            // In some cases we might need to increase the size of the base B by one, namely we require
            // K * n * t * q^2 < q * prod(B) * m_sk, where K takes into account cross terms when larger size ciphertexts
            // are used, and n is the "delta factor" for the ring. We reserve 32 bits for K * n. Here the coeff modulus
            // primes q_i are bounded to be SEAL_USER_MOD_BIT_COUNT_MAX (60) bits, and all primes in B and m_sk are
            // SEAL_INTERNAL_MOD_BIT_COUNT (61) bits.
            int total_coeff_bit_count = get_significant_bit_count_uint(q.base_prod(), q.size());

            size_t base_B_size = base_q_size;
            if (32 + t_.bit_count() + total_coeff_bit_count >=
                SEAL_INTERNAL_MOD_BIT_COUNT * safe_cast<int>(base_q_size) + SEAL_INTERNAL_MOD_BIT_COUNT)
            {
                base_B_size++;
            }

            size_t base_Bsk_size = add_safe(base_B_size, size_t(1));
            size_t base_Bsk_m_tilde_size = add_safe(base_Bsk_size, size_t(1));

            size_t base_t_gamma_size = 0;

            // Size check
            if (!product_fits_in(coeff_count_, base_Bsk_m_tilde_size))
            {
                throw logic_error("invalid parameters");
            }

            // Sample primes for B and two more primes: m_sk and gamma
            auto baseconv_primes =
                get_primes(mul_safe(size_t(2), coeff_count_), SEAL_INTERNAL_MOD_BIT_COUNT, base_Bsk_m_tilde_size);
            auto baseconv_primes_iter = baseconv_primes.cbegin();
            m_sk_ = *baseconv_primes_iter++;
            gamma_ = *baseconv_primes_iter++;
            vector<Modulus> base_B_primes;
            copy_n(baseconv_primes_iter, base_B_size, back_inserter(base_B_primes));

            // Set m_tilde_ to a non-prime value
            m_tilde_ = uint64_t(1) << 32;

            // Populate the base arrays
            base_q_ = allocate<RNSBase>(pool_, q, pool_);
            base_B_ = allocate<RNSBase>(pool_, base_B_primes, pool_);
            base_Bsk_ = allocate<RNSBase>(pool_, base_B_->extend(m_sk_));
            base_Bsk_m_tilde_ = allocate<RNSBase>(pool_, base_Bsk_->extend(m_tilde_));

            // Set up t-gamma base if t_ is non-zero (using BFV)
            if (!t_.is_zero())
            {
                base_t_gamma_size = 2;
                base_t_gamma_ = allocate<RNSBase>(pool_, vector<Modulus>{ t_, gamma_ }, pool_);
            }

            // Generate the Bsk NTTTables; these are used for NTT after base extension to Bsk
            try
            {
                CreateNTTTables(
                    coeff_count_power, vector<Modulus>(base_Bsk_->base(), base_Bsk_->base() + base_Bsk_size),
                    base_Bsk_ntt_tables_, pool_);
            }
            catch (const logic_error &)
            {
                throw logic_error("invalid rns bases");
            }

            if (!t_.is_zero())
            {
                // Set up BaseConvTool for q --> {t}
                base_q_to_t_conv_ = allocate<BaseConverter>(pool_, *base_q_, RNSBase({ t_ }, pool_), pool_);
            }

            // Set up BaseConverter for q --> Bsk
            base_q_to_Bsk_conv_ = allocate<BaseConverter>(pool_, *base_q_, *base_Bsk_, pool_);

            // Set up BaseConverter for q --> {m_tilde}
            base_q_to_m_tilde_conv_ = allocate<BaseConverter>(pool_, *base_q_, RNSBase({ m_tilde_ }, pool_), pool_);

            // Set up BaseConverter for B --> q
            base_B_to_q_conv_ = allocate<BaseConverter>(pool_, *base_B_, *base_q_, pool_);

            // Set up BaseConverter for B --> {m_sk}
            base_B_to_m_sk_conv_ = allocate<BaseConverter>(pool_, *base_B_, RNSBase({ m_sk_ }, pool_), pool_);

            if (base_t_gamma_)
            {
                // Set up BaseConverter for q --> {t, gamma}
                base_q_to_t_gamma_conv_ = allocate<BaseConverter>(pool_, *base_q_, *base_t_gamma_, pool_);
            }

            // Compute prod(B) mod q
            prod_B_mod_q_ = allocate_uint(base_q_size, pool_);
            SEAL_ITERATE(iter(prod_B_mod_q_, base_q_->base()), base_q_size, [&](auto I) {
                get<0>(I) = modulo_uint(base_B_->base_prod(), base_B_size, get<1>(I));
            });

            uint64_t temp;

            // Compute prod(q)^(-1) mod Bsk
            inv_prod_q_mod_Bsk_ = allocate<MultiplyUIntModOperand>(base_Bsk_size, pool_);
            for (size_t i = 0; i < base_Bsk_size; i++)
            {
                temp = modulo_uint(base_q_->base_prod(), base_q_size, (*base_Bsk_)[i]);
                if (!try_invert_uint_mod(temp, (*base_Bsk_)[i], temp))
                {
                    throw logic_error("invalid rns bases");
                }
                inv_prod_q_mod_Bsk_[i].set(temp, (*base_Bsk_)[i]);
            }

            // Compute prod(B)^(-1) mod m_sk
            temp = modulo_uint(base_B_->base_prod(), base_B_size, m_sk_);
            if (!try_invert_uint_mod(temp, m_sk_, temp))
            {
                throw logic_error("invalid rns bases");
            }
            inv_prod_B_mod_m_sk_.set(temp, m_sk_);

            // Compute m_tilde^(-1) mod Bsk
            inv_m_tilde_mod_Bsk_ = allocate<MultiplyUIntModOperand>(base_Bsk_size, pool_);
            SEAL_ITERATE(iter(inv_m_tilde_mod_Bsk_, base_Bsk_->base()), base_Bsk_size, [&](auto I) {
                if (!try_invert_uint_mod(barrett_reduce_64(m_tilde_.value(), get<1>(I)), get<1>(I), temp))
                {
                    throw logic_error("invalid rns bases");
                }
                get<0>(I).set(temp, get<1>(I));
            });

            // Compute prod(q)^(-1) mod m_tilde
            temp = modulo_uint(base_q_->base_prod(), base_q_size, m_tilde_);
            if (!try_invert_uint_mod(temp, m_tilde_, temp))
            {
                throw logic_error("invalid rns bases");
            }
            neg_inv_prod_q_mod_m_tilde_.set(negate_uint_mod(temp, m_tilde_), m_tilde_);

            // Compute prod(q) mod Bsk
            prod_q_mod_Bsk_ = allocate_uint(base_Bsk_size, pool_);
            SEAL_ITERATE(iter(prod_q_mod_Bsk_, base_Bsk_->base()), base_Bsk_size, [&](auto I) {
                get<0>(I) = modulo_uint(base_q_->base_prod(), base_q_size, get<1>(I));
            });

            if (base_t_gamma_)
            {
                // Compute gamma^(-1) mod t
                if (!try_invert_uint_mod(barrett_reduce_64(gamma_.value(), t_), t_, temp))
                {
                    throw logic_error("invalid rns bases");
                }
                inv_gamma_mod_t_.set(temp, t_);

                // Compute prod({t, gamma}) mod q
                prod_t_gamma_mod_q_ = allocate<MultiplyUIntModOperand>(base_q_size, pool_);
                SEAL_ITERATE(iter(prod_t_gamma_mod_q_, base_q_->base()), base_q_size, [&](auto I) {
                    get<0>(I).set(
                        multiply_uint_mod((*base_t_gamma_)[0].value(), (*base_t_gamma_)[1].value(), get<1>(I)),
                        get<1>(I));
                });

                // Compute -prod(q)^(-1) mod {t, gamma}
                neg_inv_q_mod_t_gamma_ = allocate<MultiplyUIntModOperand>(base_t_gamma_size, pool_);
                SEAL_ITERATE(iter(neg_inv_q_mod_t_gamma_, base_t_gamma_->base()), base_t_gamma_size, [&](auto I) {
                    get<0>(I).operand = modulo_uint(base_q_->base_prod(), base_q_size, get<1>(I));
                    if (!try_invert_uint_mod(get<0>(I).operand, get<1>(I), get<0>(I).operand))
                    {
                        throw logic_error("invalid rns bases");
                    }
                    get<0>(I).set(negate_uint_mod(get<0>(I).operand, get<1>(I)), get<1>(I));
                });
            }

            // Compute q[last]^(-1) mod q[i] for i = 0..last-1
            // This is used by modulus switching and rescaling
            inv_q_last_mod_q_ = allocate<MultiplyUIntModOperand>(base_q_size - 1, pool_);
            SEAL_ITERATE(iter(inv_q_last_mod_q_, base_q_->base()), base_q_size - 1, [&](auto I) {
                if (!try_invert_uint_mod((*base_q_)[base_q_size - 1].value(), get<1>(I), temp))
                {
                    throw logic_error("invalid rns bases");
                }
                get<0>(I).set(temp, get<1>(I));
            });

            if (t_.value() != 0)
            {
                if (!try_invert_uint_mod(base_q_->base()[base_q_size - 1].value(), t_, inv_q_last_mod_t_))
                {
                    throw logic_error("invalid rns bases");
                }

                q_last_mod_t_ = barrett_reduce_64(base_q_->base()[base_q_size - 1].value(), t_);
            }
        }

        void RNSTool::divide_and_round_q_last_inplace(RNSIter input, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            CoeffIter last_input = input[base_q_size - 1];

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;
            add_poly_scalar_coeffmod(last_input, coeff_count_, half, last_modulus, last_input);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            SEAL_ITERATE(iter(input, inv_q_last_mod_q_, base_q_->base()), base_q_size - 1, [&](auto I) {
                // (ct mod qk) mod qi
                modulo_poly_coeffs(last_input, coeff_count_, get<2>(I), temp);

                // Subtract rounding correction here; the negative sign will turn into a plus in the next subtraction
                uint64_t half_mod = barrett_reduce_64(half, get<2>(I));
                sub_poly_scalar_coeffmod(temp, coeff_count_, half_mod, get<2>(I), temp);

                // (ct mod qi) - (ct mod qk) mod qi
                sub_poly_coeffmod(get<0>(I), temp, coeff_count_, get<2>(I), get<0>(I));

                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<0>(I));
            });
        }

        void RNSTool::divide_and_round_q_last_ntt_inplace(
            RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!rns_ntt_tables)
            {
                throw invalid_argument("rns_ntt_tables cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            CoeffIter last_input = input[base_q_size - 1];

            // Convert to non-NTT form
            inverse_ntt_negacyclic_harvey(last_input, rns_ntt_tables[base_q_size - 1]);

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;
            add_poly_scalar_coeffmod(last_input, coeff_count_, half, last_modulus, last_input);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            SEAL_ITERATE(iter(input, inv_q_last_mod_q_, base_q_->base(), rns_ntt_tables), base_q_size - 1, [&](auto I) {
                // (ct mod qk) mod qi
                if (get<2>(I).value() < last_modulus.value())
                {
                    modulo_poly_coeffs(last_input, coeff_count_, get<2>(I), temp);
                }
                else
                {
                    set_uint(last_input, coeff_count_, temp);
                }

                // Lazy subtraction here. ntt_negacyclic_harvey_lazy can take 0 < x < 4*qi input.
                uint64_t neg_half_mod = get<2>(I).value() - barrett_reduce_64(half, get<2>(I));

                // Note: lambda function parameter must be passed by reference here
                SEAL_ITERATE(temp, coeff_count_, [&](auto &J) { J += neg_half_mod; });
#if SEAL_USER_MOD_BIT_COUNT_MAX <= 60
                // Since SEAL uses at most 60-bit moduli, 8*qi < 2^63.
                // This ntt_negacyclic_harvey_lazy results in [0, 4*qi).
                uint64_t qi_lazy = get<2>(I).value() << 2;
                ntt_negacyclic_harvey_lazy(temp, get<3>(I));
#else
                // 2^60 < pi < 2^62, then 4*pi < 2^64, we perfrom one reduction from [0, 4*qi) to [0, 2*qi) after ntt.
                uint64_t qi_lazy = get<2>(I).value() << 1;
                ntt_negacyclic_harvey_lazy(temp, get<3>(I));

                // Note: lambda function parameter must be passed by reference here
                SEAL_ITERATE(temp, coeff_count_, [&](auto &J) {
                    J -= (qi_lazy & static_cast<uint64_t>(-static_cast<int64_t>(J >= qi_lazy)));
                });
#endif
                // Lazy subtraction again, results in [0, 2*qi_lazy),
                // The reduction [0, 2*qi_lazy) -> [0, qi) is done implicitly in multiply_poly_scalar_coeffmod.
                SEAL_ITERATE(iter(get<0>(I), temp), coeff_count_, [&](auto J) { get<0>(J) += qi_lazy - get<1>(J); });

                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<0>(I));
            });
        }

        void RNSTool::fastbconv_sk(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base Bsk
            Ensure: Output in base q
            */

            size_t base_q_size = base_q_->size();
            size_t base_B_size = base_B_->size();

            // Fast convert B -> q; input is in Bsk but we only use B
            base_B_to_q_conv_->fast_convert_array(input, destination, pool);

            // Compute alpha_sk
            // Fast convert B -> {m_sk}; input is in Bsk but we only use B
            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            base_B_to_m_sk_conv_->fast_convert_array(input, RNSIter(temp, coeff_count_), pool);

            // Take the m_sk part of input, subtract from temp, and multiply by inv_prod_B_mod_m_sk_
            // Note: input_sk is allocated in input[base_B_size]
            SEAL_ALLOCATE_GET_COEFF_ITER(alpha_sk, coeff_count_, pool);
#ifdef __ARM_FEATURE_SVE
            {
                uint64_t *alpha_ptr = alpha_sk;
                const uint64_t *temp_ptr = temp;
                const uint64_t *in_sk_ptr = input[base_B_size].ptr();

                svuint64_t sv_m_sk = svdup_n_u64(m_sk_.value());
                svuint64_t sv_inv_op = svdup_n_u64(inv_prod_B_mod_m_sk_.operand);
                svuint64_t sv_inv_q = svdup_n_u64(inv_prod_B_mod_m_sk_.quotient);

                // alpha_sk = (temp + (m_sk - input_sk)) * inv_prod_B_mod_m_sk_ mod m_sk,
                // processed N coefficients per iteration.
                const uint64_t N = svcntd();
                const svbool_t ptrue = svptrue_b64();
                const size_t full = (coeff_count_ / N) * N;
                size_t j = 0;
                for (; j < full; j += N)
                {
                    svuint64_t v_temp = svld1_u64(ptrue, temp_ptr + j);
                    svuint64_t v_in_sk = svld1_u64(ptrue, in_sk_ptr + j);
                    svuint64_t v_x = svadd_u64_x(ptrue, v_temp, svsub_u64_x(ptrue, sv_m_sk, v_in_sk));
                    svuint64_t v_hi = sve32_mulhi_u64(ptrue, v_x, sv_inv_q);
                    svuint64_t v_lo = svmul_u64_x(ptrue, v_x, sv_inv_op);
                    v_lo = svmls_u64_x(ptrue, v_lo, sv_m_sk, v_hi);
                    svbool_t ge = svcmpge_u64(ptrue, v_lo, sv_m_sk);
                    v_lo = svsub_u64_m(ge, v_lo, sv_m_sk);
                    svst1_u64(ptrue, alpha_ptr + j, v_lo);
                }
                if (j < coeff_count_)
                {
                    svbool_t pg = svwhilelt_b64(j, coeff_count_);
                    svuint64_t v_temp = svld1_u64(pg, temp_ptr + j);
                    svuint64_t v_in_sk = svld1_u64(pg, in_sk_ptr + j);
                    svuint64_t v_x = svadd_u64_z(pg, v_temp, svsub_u64_z(pg, sv_m_sk, v_in_sk));
                    svuint64_t v_hi = sve32_mulhi_u64(pg, v_x, sv_inv_q);
                    svuint64_t v_lo = svmul_u64_x(pg, v_x, sv_inv_op);
                    v_lo = svmls_u64_x(pg, v_lo, sv_m_sk, v_hi);
                    svbool_t ge = svcmpge_u64(pg, v_lo, sv_m_sk);
                    v_lo = svsub_u64_m(ge, v_lo, sv_m_sk);
                    svst1_u64(pg, alpha_ptr + j, v_lo);
                }
            }
#else
            SEAL_ITERATE(iter(alpha_sk, temp, input[base_B_size]), coeff_count_, [&](auto I) {
                // It is not necessary for the negation to be reduced modulo the small prime
                get<0>(I) = multiply_uint_mod(get<1>(I) + (m_sk_.value() - get<2>(I)), inv_prod_B_mod_m_sk_, m_sk_);
            });
#endif

            // alpha_sk is now ready for the Shenoy-Kumaresan conversion; however, note that our
            // alpha_sk here is not a centered reduction, so we need to apply a correction below.
            const uint64_t m_sk_div_2 = m_sk_.value() >> 1;
#ifdef __ARM_FEATURE_SVE
            // SVE vectorized fastbconv_sk correction loop
            for (size_t i = 0; i < base_q_size; i++)
            {
                // Set up multiplication helpers (matching original code)
                MultiplyUIntModOperand local_prod;
                local_prod.set(prod_B_mod_q_[i], base_q_->base()[i]);

                MultiplyUIntModOperand local_neg_prod;
                local_neg_prod.set(base_q_->base()[i].value() - prod_B_mod_q_[i], base_q_->base()[i]);

                const uint64_t *alpha_ptr = alpha_sk;
                uint64_t *dest_ptr = destination[i].ptr();
                const uint64_t mod_val = base_q_->base()[i].value();

                // SVE constants
                svuint64_t sv_mod = svdup_n_u64(mod_val);
                svuint64_t sv_m_sk = svdup_n_u64(m_sk_.value());
                svuint64_t sv_div_2 = svdup_n_u64(m_sk_div_2);
                svuint64_t sv_prod_op = svdup_n_u64(local_prod.operand);
                svuint64_t sv_prod_q = svdup_n_u64(local_prod.quotient);
                svuint64_t sv_neg_op = svdup_n_u64(local_neg_prod.operand);
                svuint64_t sv_neg_q = svdup_n_u64(local_neg_prod.quotient);

                const uint64_t N = svcntd();
                const svbool_t ptrue = svptrue_b64();
                const size_t full = (coeff_count_ / N) * N;
                size_t j = 0;
                for (; j < full; j += N)
                {
                    svuint64_t v_alpha = svld1_u64(ptrue, alpha_ptr + j);
                    svbool_t need_neg = svcmpgt_u64(ptrue, v_alpha, sv_div_2);
                    svuint64_t v_dest = svld1_u64(ptrue, dest_ptr + j);
                    svuint64_t v_neg_alpha = svsub_u64_x(ptrue, sv_m_sk, v_alpha);
                    svuint64_t v_neg_hi = sve32_mulhi_u64(ptrue, v_neg_alpha, sv_prod_q);
                    svuint64_t v_neg_lo = svmul_u64_x(ptrue, v_neg_alpha, sv_prod_op);
                    v_neg_lo = svmls_u64_x(ptrue, v_neg_lo, sv_mod, v_neg_hi);
                    svuint64_t v_neg_result = svadd_u64_x(ptrue, v_neg_lo, v_dest);
                    svuint64_t v_pos_hi = sve32_mulhi_u64(ptrue, v_alpha, sv_neg_q);
                    svuint64_t v_pos_lo = svmul_u64_x(ptrue, v_alpha, sv_neg_op);
                    v_pos_lo = svmls_u64_x(ptrue, v_pos_lo, sv_mod, v_pos_hi);
                    svuint64_t v_pos_result = svadd_u64_x(ptrue, v_pos_lo, v_dest);
                    svuint64_t v_result = svsel_u64(need_neg, v_neg_result, v_pos_result);
                    svbool_t ge = svcmpge_u64(ptrue, v_result, sv_mod);
                    v_result = svsub_u64_m(ge, v_result, sv_mod);
                    svst1_u64(ptrue, dest_ptr + j, v_result);
                }
                if (j < coeff_count_)
                {
                    svbool_t pg = svwhilelt_b64(j, coeff_count_);
                    svuint64_t v_alpha = svld1_u64(pg, alpha_ptr + j);
                    svbool_t need_neg = svcmpgt_u64(pg, v_alpha, sv_div_2);
                    svuint64_t v_dest = svld1_u64(pg, dest_ptr + j);
                    svuint64_t v_neg_alpha = svsub_u64_z(pg, sv_m_sk, v_alpha);
                    svuint64_t v_neg_hi = sve32_mulhi_u64(pg, v_neg_alpha, sv_prod_q);
                    svuint64_t v_neg_lo = svmul_u64_x(pg, v_neg_alpha, sv_prod_op);
                    v_neg_lo = svmls_u64_x(pg, v_neg_lo, sv_mod, v_neg_hi);
                    svuint64_t v_neg_result = svadd_u64_z(pg, v_neg_lo, v_dest);
                    svuint64_t v_pos_hi = sve32_mulhi_u64(pg, v_alpha, sv_neg_q);
                    svuint64_t v_pos_lo = svmul_u64_x(pg, v_alpha, sv_neg_op);
                    v_pos_lo = svmls_u64_x(pg, v_pos_lo, sv_mod, v_pos_hi);
                    svuint64_t v_pos_result = svadd_u64_z(pg, v_pos_lo, v_dest);
                    svuint64_t v_result = svsel_u64(need_neg, v_neg_result, v_pos_result);
                    svbool_t ge = svcmpge_u64(pg, v_result, sv_mod);
                    v_result = svsub_u64_m(ge, v_result, sv_mod);
                    svst1_u64(pg, dest_ptr + j, v_result);
                }
            }
#else
            SEAL_ITERATE(iter(prod_B_mod_q_, base_q_->base(), destination), base_q_size, [&](auto I) {
                // Set up the multiplication helpers
                MultiplyUIntModOperand prod_B_mod_q_elt;
                prod_B_mod_q_elt.set(get<0>(I), get<1>(I));

                MultiplyUIntModOperand neg_prod_B_mod_q_elt;
                neg_prod_B_mod_q_elt.set(get<1>(I).value() - get<0>(I), get<1>(I));

                SEAL_ITERATE(iter(alpha_sk, get<2>(I)), coeff_count_, [&](auto J) {
                    // Correcting alpha_sk since it represents a negative value
                    if (get<0>(J) > m_sk_div_2)
                    {
                        get<1>(J) = multiply_add_uint_mod(
                            negate_uint_mod(get<0>(J), m_sk_), prod_B_mod_q_elt, get<1>(J), get<1>(I));
                    }
                    // No correction needed
                    else
                    {
                        // It is not necessary for the negation to be reduced modulo the small prime
                        get<1>(J) = multiply_add_uint_mod(get<0>(J), neg_prod_B_mod_q_elt, get<1>(J), get<1>(I));
                    }
                });
            });
#endif
        }

        void RNSTool::sm_mrq(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base Bsk U {m_tilde}
            Ensure: Output in base Bsk
            */

            size_t base_Bsk_size = base_Bsk_->size();

            // The last component of the input is mod m_tilde
            ConstCoeffIter input_m_tilde = input[base_Bsk_size];
            const uint64_t m_tilde_div_2 = m_tilde_.value() >> 1;

            // Compute r_m_tilde
            SEAL_ALLOCATE_GET_COEFF_ITER(r_m_tilde, coeff_count_, pool);
            multiply_poly_scalar_coeffmod(
                input_m_tilde, coeff_count_, neg_inv_prod_q_mod_m_tilde_, m_tilde_, r_m_tilde);

#ifdef __ARM_FEATURE_SVE
            // SVE vectorized sm_mrq: avoid gather-loads by iterating over input RNS components
            // and using contiguous loads + scalar gather from r_m_tilde / destination
            for (size_t i = 0; i < base_Bsk_size; i++)
            {
                // Set up the multiplication helpers (matching the original SEAL_ITERATE)
                MultiplyUIntModOperand local_prod_q;
                local_prod_q.set(prod_q_mod_Bsk_[i], base_Bsk_->base()[i]);

                const uint64_t *in_ptr = input[i].ptr();
                uint64_t *dest_ptr = destination[i].ptr();
                const uint64_t *r_m_tilde_ptr = r_m_tilde;
                const uint64_t mod_val = base_Bsk_->base()[i].value();

                // SVE constants
                svuint64_t sv_mod = svdup_n_u64(mod_val);
                svuint64_t sv_inv_m_tilde = svdup_n_u64(inv_m_tilde_mod_Bsk_[i].operand);
                svuint64_t sv_inv_m_tilde_q = svdup_n_u64(inv_m_tilde_mod_Bsk_[i].quotient);
                svuint64_t sv_prod_q = svdup_n_u64(local_prod_q.operand);
                svuint64_t sv_prod_q_q = svdup_n_u64(local_prod_q.quotient);
                svuint64_t sv_div_2 = svdup_n_u64(m_tilde_div_2);
                svuint64_t sv_fix_val = svdup_n_u64(mod_val - m_tilde_.value());
                svuint64_t sv_zero = svdup_n_u64(0);

                const uint64_t N = svcntd();
                const svbool_t ptrue = svptrue_b64();
                const size_t full = (coeff_count_ / N) * N;
                size_t j = 0;
                for (; j < full; j += N)
                {
                    svuint64_t v_rmt = svld1_u64(ptrue, r_m_tilde_ptr + j);
                    svbool_t need_fix = svcmpge_u64(ptrue, v_rmt, sv_div_2);
                    svuint64_t v_fix = svsel_u64(need_fix, sv_fix_val, sv_zero);
                    v_rmt = svadd_u64_x(ptrue, v_rmt, v_fix);
                    svuint64_t v_in = svld1_u64(ptrue, in_ptr + j);
                    svuint64_t v_prod_hi = sve32_mulhi_u64(ptrue, v_rmt, sv_prod_q_q);
                    svuint64_t v_prod_lo = svmul_u64_x(ptrue, v_rmt, sv_prod_q);
                    svuint64_t v_ma = svmls_u64_x(ptrue, v_prod_lo, sv_mod, v_prod_hi);
                    svuint64_t v_sum = svadd_u64_x(ptrue, v_ma, v_in);
                    svuint64_t v_sum_hi = sve32_mulhi_u64(ptrue, v_sum, sv_inv_m_tilde_q);
                    svuint64_t v_sum_lo = svmul_u64_x(ptrue, v_sum, sv_inv_m_tilde);
                    v_sum_lo = svmls_u64_x(ptrue, v_sum_lo, sv_mod, v_sum_hi);
                    svbool_t ge = svcmpge_u64(ptrue, v_sum_lo, sv_mod);
                    v_sum_lo = svsub_u64_m(ge, v_sum_lo, sv_mod);
                    svst1_u64(ptrue, dest_ptr + j, v_sum_lo);
                }
                if (j < coeff_count_)
                {
                    svbool_t pg = svwhilelt_b64(j, coeff_count_);
                    svuint64_t v_rmt = svld1_u64(pg, r_m_tilde_ptr + j);
                    svbool_t need_fix = svcmpge_u64(pg, v_rmt, sv_div_2);
                    svuint64_t v_fix = svsel_u64(need_fix, sv_fix_val, sv_zero);
                    v_rmt = svadd_u64_z(pg, v_rmt, v_fix);
                    svuint64_t v_in = svld1_u64(pg, in_ptr + j);
                    svuint64_t v_prod_hi = sve32_mulhi_u64(pg, v_rmt, sv_prod_q_q);
                    svuint64_t v_prod_lo = svmul_u64_x(pg, v_rmt, sv_prod_q);
                    svuint64_t v_ma = svmls_u64_x(pg, v_prod_lo, sv_mod, v_prod_hi);
                    svuint64_t v_sum = svadd_u64_z(pg, v_ma, v_in);
                    svuint64_t v_sum_hi = sve32_mulhi_u64(pg, v_sum, sv_inv_m_tilde_q);
                    svuint64_t v_sum_lo = svmul_u64_x(pg, v_sum, sv_inv_m_tilde);
                    v_sum_lo = svmls_u64_x(pg, v_sum_lo, sv_mod, v_sum_hi);
                    svbool_t ge = svcmpge_u64(pg, v_sum_lo, sv_mod);
                    v_sum_lo = svsub_u64_m(ge, v_sum_lo, sv_mod);
                    svst1_u64(pg, dest_ptr + j, v_sum_lo);
                }
            }
#else
            SEAL_ITERATE(
                iter(input, prod_q_mod_Bsk_, inv_m_tilde_mod_Bsk_, base_Bsk_->base(), destination), base_Bsk_size,
                [&](auto I) {
                    MultiplyUIntModOperand prod_q_mod_Bsk_elt;
                    prod_q_mod_Bsk_elt.set(get<1>(I), get<3>(I));
                    SEAL_ITERATE(iter(get<0>(I), r_m_tilde, get<4>(I)), coeff_count_, [&](auto J) {
                        // We need centered reduction of r_m_tilde modulo Bsk. Note that m_tilde is chosen
                        // to be a power of two so we have '>=' below.
                        uint64_t temp = get<1>(J);
                        if (temp >= m_tilde_div_2)
                        {
                            temp += get<3>(I).value() - m_tilde_.value();
                        }

                        // Compute (input + q*r_m_tilde)*m_tilde^(-1) mod Bsk
                        get<2>(J) = multiply_uint_mod(
                            multiply_add_uint_mod(temp, prod_q_mod_Bsk_elt, get<0>(J), get<3>(I)), get<2>(I),
                            get<3>(I));
                    });
                });
#endif
        }

        void RNSTool::fast_floor(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base q U Bsk
            Ensure: Output in base Bsk
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();

            // Convert q -> Bsk
            base_q_to_Bsk_conv_->fast_convert_array(input, destination, pool);

            // Move input pointer to past the base q components
            input += base_q_size;
#ifdef __ARM_FEATURE_SVE
            for (size_t i = 0; i < base_Bsk_size; i++)
            {
                const uint64_t *in_ptr = input[i].ptr();
                uint64_t *dest_ptr = destination[i].ptr();
                const uint64_t mod_val = base_Bsk_->base()[i].value();

                svuint64_t sv_mod = svdup_n_u64(mod_val);
                svuint64_t sv_inv_op = svdup_n_u64(inv_prod_q_mod_Bsk_[i].operand);
                svuint64_t sv_inv_q = svdup_n_u64(inv_prod_q_mod_Bsk_[i].quotient);

                const uint64_t N = svcntd();
                const svbool_t ptrue = svptrue_b64();
                const size_t full = (coeff_count_ / N) * N;
                size_t j = 0;
                for (; j < full; j += N)
                {
                    svuint64_t v_in = svld1_u64(ptrue, in_ptr + j);
                    svuint64_t v_dest = svld1_u64(ptrue, dest_ptr + j);
                    svuint64_t v_x = svadd_u64_x(ptrue, v_in, svsub_u64_x(ptrue, sv_mod, v_dest));
                    svuint64_t v_hi = sve32_mulhi_u64(ptrue, v_x, sv_inv_q);
                    svuint64_t v_lo = svmul_u64_x(ptrue, v_x, sv_inv_op);
                    v_lo = svmls_u64_x(ptrue, v_lo, sv_mod, v_hi);
                    svbool_t ge = svcmpge_u64(ptrue, v_lo, sv_mod);
                    v_lo = svsub_u64_m(ge, v_lo, sv_mod);
                    svst1_u64(ptrue, dest_ptr + j, v_lo);
                }
                if (j < coeff_count_)
                {
                    svbool_t pg = svwhilelt_b64(j, coeff_count_);
                    svuint64_t v_in = svld1_u64(pg, in_ptr + j);
                    svuint64_t v_dest = svld1_u64(pg, dest_ptr + j);
                    svuint64_t v_x = svadd_u64_z(pg, v_in, svsub_u64_z(pg, sv_mod, v_dest));
                    svuint64_t v_hi = sve32_mulhi_u64(pg, v_x, sv_inv_q);
                    svuint64_t v_lo = svmul_u64_x(pg, v_x, sv_inv_op);
                    v_lo = svmls_u64_x(pg, v_lo, sv_mod, v_hi);
                    svbool_t ge = svcmpge_u64(pg, v_lo, sv_mod);
                    v_lo = svsub_u64_m(ge, v_lo, sv_mod);
                    svst1_u64(pg, dest_ptr + j, v_lo);
                }
            }
#else
            SEAL_ITERATE(iter(input, inv_prod_q_mod_Bsk_, base_Bsk_->base(), destination), base_Bsk_size, [&](auto I) {
                SEAL_ITERATE(iter(get<0>(I), get<3>(I)), coeff_count_, [&](auto J) {
                    // It is not necessary for the negation to be reduced modulo base_Bsk_elt
                    get<1>(J) = multiply_uint_mod(get<0>(J) + (get<2>(I).value() - get<1>(J)), get<1>(I), get<2>(I));
                });
            });
#endif
        }

        void RNSTool::fastbconv_m_tilde(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in q
            Ensure: Output in Bsk U {m_tilde}
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();

            // We need to multiply first the input with m_tilde mod q
            // This is to facilitate Montgomery reduction in the next step of multiplication
            // This is NOT an ideal approach: as mentioned in BEHZ16, multiplication by
            // m_tilde can be easily merge into the base conversion operation; however, then
            // we could not use the BaseConverter as below without modifications.
            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count_, base_q_size, pool);
            multiply_poly_scalar_coeffmod(input, base_q_size, m_tilde_.value(), base_q_->base(), temp);

            // Now convert to Bsk
            base_q_to_Bsk_conv_->fast_convert_array(temp, destination, pool);

            // Finally convert to {m_tilde}
            base_q_to_m_tilde_conv_->fast_convert_array(temp, destination + base_Bsk_size, pool);
        }

        void RNSTool::decrypt_scale_and_round(ConstRNSIter input, CoeffIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            size_t base_t_gamma_size = base_t_gamma_->size();

            // Compute |gamma * t|_qi * ct(s)
            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count_, base_q_size, pool);
            SEAL_ITERATE(iter(input, prod_t_gamma_mod_q_, base_q_->base(), temp), base_q_size, [&](auto I) {
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<3>(I));
            });

            // Make another temp destination to get the poly in mod {t, gamma}
            SEAL_ALLOCATE_GET_RNS_ITER(temp_t_gamma, coeff_count_, base_t_gamma_size, pool);

            // Convert from q to {t, gamma}
            base_q_to_t_gamma_conv_->fast_convert_array(temp, temp_t_gamma, pool);

            // Multiply by -prod(q)^(-1) mod {t, gamma}
            SEAL_ITERATE(
                iter(temp_t_gamma, neg_inv_q_mod_t_gamma_, base_t_gamma_->base(), temp_t_gamma), base_t_gamma_size,
                [&](auto I) {
                    multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<3>(I));
                });

            // Need to correct values in temp_t_gamma (gamma component only) which are
            // larger than floor(gamma/2)
            uint64_t gamma_div_2 = (*base_t_gamma_)[1].value() >> 1;

            // Now compute the subtraction to remove error and perform final multiplication by
            // gamma inverse mod t
            SEAL_ITERATE(iter(temp_t_gamma[0], temp_t_gamma[1], destination), coeff_count_, [&](auto I) {
                // Need correction because of centered mod
                if (get<1>(I) > gamma_div_2)
                {
                    // Compute -(gamma - a) instead of (a - gamma)
                    get<2>(I) = add_uint_mod(get<0>(I), barrett_reduce_64(gamma_.value() - get<1>(I), t_), t_);
                }
                // No correction needed
                else
                {
                    get<2>(I) = sub_uint_mod(get<0>(I), barrett_reduce_64(get<1>(I), t_), t_);
                }

                // If this coefficient was non-zero, multiply by gamma^(-1)
                if (0 != get<2>(I))
                {
                    // Perform final multiplication by gamma inverse mod t
                    get<2>(I) = multiply_uint_mod(get<2>(I), inv_gamma_mod_t_, t_);
                }
            });
        }

        void RNSTool::mod_t_and_divide_q_last_inplace(RNSIter input, MemoryPoolHandle pool) const
        {
            size_t modulus_size = base_q_->size();
            const Modulus *curr_modulus = base_q_->base();
            const Modulus plain_modulus = t_;
            uint64_t last_modulus_value = curr_modulus[modulus_size - 1].value();

            SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(neg_c_last_mod_t, coeff_count_, pool);
            // neg_c_last_mod_t = - c_last (mod t)
            modulo_poly_coeffs(CoeffIter(input[modulus_size - 1]), coeff_count_, plain_modulus, neg_c_last_mod_t);
            negate_poly_coeffmod(neg_c_last_mod_t, coeff_count_, plain_modulus, neg_c_last_mod_t);
            if (inv_q_last_mod_t_ != 1)
            {
                // neg_c_last_mod_t *= q_last^(-1) (mod t)
                multiply_poly_scalar_coeffmod(
                    neg_c_last_mod_t, coeff_count_, inv_q_last_mod_t_, plain_modulus, neg_c_last_mod_t);
            }

            SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(delta_mod_q_i, coeff_count_, pool);

            SEAL_ITERATE(iter(input, curr_modulus, inv_q_last_mod_q_), modulus_size - 1, [&](auto I) {
                // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
                modulo_poly_coeffs(neg_c_last_mod_t, coeff_count_, get<1>(I), delta_mod_q_i);

                // delta_mod_q_i *= q_last (mod q_i)
                multiply_poly_scalar_coeffmod(
                    delta_mod_q_i, coeff_count_, last_modulus_value, get<1>(I), delta_mod_q_i);

                // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
                const uint64_t two_times_q_i = get<1>(I).value() << 1;
                SEAL_ITERATE(iter(get<0>(I), delta_mod_q_i, input[modulus_size - 1]), coeff_count_, [&](auto J) {
                    get<0>(J) += two_times_q_i - barrett_reduce_64(get<2>(J), get<1>(I)) - get<1>(J);
                });

                // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<2>(I), get<1>(I), get<0>(I));
            });
        }

        void RNSTool::decrypt_modt(RNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const
        {
            // Use exact base convension rather than convert the base through the compose API
            base_q_to_t_conv_->exact_convert_array(phase, destination, pool);
        }
    } // namespace util
} // namespace seal
