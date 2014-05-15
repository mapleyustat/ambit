/*
 * Copyright (C) 2013 Devin Matthews
 *
 * This is a slimmed down version of the tensor framework developed by
 * Devin Matthews. The version by Devin was tied to Aquarius. This
 * version is not.
 *
 * Copyright (C) 2013  Justin Turney
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "dense_tensor.h"
#include <cassert>

namespace ambit { namespace tensor {

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, T val)
    : local_tensor<dense_tensor<T>,T>(name, val) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const dense_tensor<T>& A, T val)
    : local_tensor< dense_tensor<T>,T >(name, val) {}

template <typename T>
dense_tensor<T>::dense_tensor(const dense_tensor<T>& A)
    : local_tensor< dense_tensor<T>,T >(A) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const dense_tensor<T>& A)
    : local_tensor< dense_tensor<T>,T >(name, A) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, dense_tensor<T>& A, typename local_tensor<dense_tensor<T>,T>::CopyType type)
    : local_tensor< dense_tensor<T>,T >(name, A, type) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const std::vector<int>& len, T* data, bool zero)
    : local_tensor< dense_tensor<T>,T >(name, len, std::vector<int>(), get_size(ndim, len, std::vector<int>()), data, zero) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const std::vector<int>& len, bool zero)
    : local_tensor< dense_tensor<T>,T >(name, len, std::vector<int>(), get_size(ndim, len, std::vector<int>()), zero) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld, T* data, bool zero)
    : local_tensor< dense_tensor<T>,T >(name, len, ld, get_size(ndim, len, ld), data, zero) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld, bool zero)
    : local_tensor< dense_tensor<T>,T >(name, len, ld, get_size(ndim, len, ld), zero) {}

template <typename T>
dense_tensor<T>::dense_tensor(const std::string& name, const std::string& indices)
    : local_tensor<dense_tensor<T>, T>(name, indices) {}

template <typename T>
uint64_t dense_tensor<T>::get_size(int ndim, const std::vector<int>& len, const std::vector<int>& ld)
{
    int64_t r = tensor_size_dense(ndim, len.data(), (ld.size() == 0 ? NULL : ld.data()));

    #ifdef VALIDATE_INPUTS
    CHECK_RETURN_VALUE(r);
    #endif //VALIDATE_INPUTS

    return r;
}

template <typename T>
void dense_tensor<T>::print() const
{
    printf("Name: %s\n", get_name().c_str());
    CHECK_RETURN_VALUE(
    tensor_print_dense(data, ndim, len.data(), ld.data()));
}

template <typename T>
void dense_tensor<T>::mult(const T alpha, const dense_tensor<T>& A, const std::string& idx_A,
                                          const dense_tensor<T>& B, const std::string& idx_B,
                          const T beta,                             const std::string& idx_C)
{
    std::vector<int> idx_A_(    A.ndim);
    std::vector<int> idx_B_(    B.ndim);
    std::vector<int> idx_C_(this->ndim);

    for (int i = 0;i <     A.ndim;i++) idx_A_[i] = idx_A[i];
    for (int i = 0;i <     B.ndim;i++) idx_B_[i] = idx_B[i];
    for (int i = 0;i < this->ndim;i++) idx_C_[i] = idx_C[i];

    CHECK_RETURN_VALUE(
    tensor_mult_dense_(alpha, A.data, A.ndim, A.len.data(), A.ld.data(), idx_A_.data(),
                              B.data, B.ndim, B.len.data(), B.ld.data(), idx_B_.data(),
                       beta,    data,   ndim,   len.data(),   ld.data(), idx_C_.data()));
}

template <typename T>
void dense_tensor<T>::sum(const T alpha, const dense_tensor<T>& A, const std::string& idx_A,
                         const T beta,                           const std::string& idx_B)
{
    std::vector<int> idx_A_(    A.ndim);
    std::vector<int> idx_B_(this->ndim);

    for (int i = 0;i <     A.ndim;i++) idx_A_[i] = idx_A[i];
    for (int i = 0;i < this->ndim;i++) idx_B_[i] = idx_B[i];

    CHECK_RETURN_VALUE(
    tensor_sum_dense_(alpha, A.data, A.ndim, A.len.data(), A.ld.data(), idx_A_.data(),
                      beta,    data,   ndim,   len.data(),   ld.data(), idx_B_.data()));
}

template <typename T>
void dense_tensor<T>::scale(const T alpha, const std::string& idx_A)
{
    std::vector<int> idx_A_(this->ndim);

    for (int i = 0;i < this->ndim;i++) idx_A_[i] = idx_A[i];

    CHECK_RETURN_VALUE(
    tensor_scale_dense_(alpha, data, ndim, len.data(), ld.data(), idx_A_.data()));
}

INSTANTIATE_SPECIALIZATIONS(dense_tensor);

}
}
