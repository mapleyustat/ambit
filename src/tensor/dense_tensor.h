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

#if !defined(AMBIT_LIB_TENSOR_DENSE_TENSOR)
#define AMBIT_LIB_TENSOR_DENSE_TENSOR

#include "local_tensor.h"
#include <vector>

namespace ambit {

namespace tensor {

template <typename T>
struct DenseTensor : public LocalTensor< DenseTensor<T>, T>
{
    INHERIT_FROM_LOCAL_TENSOR(DenseTensor<T>,T)

    typedef typename LocalTensor<DenseTensor<T>,T>::CopyType CopyType_;

public:
    DenseTensor(const std::string& name, T val = (T)0);
    DenseTensor(const std::string& name, const DenseTensor<T>& A, T val);
    DenseTensor(const DenseTensor<T>& A);
    DenseTensor(const std::string& name, const DenseTensor<T>& A);
    DenseTensor(const std::string& name, DenseTensor<T>& A, CopyType_ type=CLONE);
    DenseTensor(const std::string& name, const std::vector<int>& len, T* data, bool zero=false);
    DenseTensor(const std::string& name, const std::vector<int>& len, bool zero=true);
    DenseTensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld, T* data, bool zero=false);
    DenseTensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld, bool zero=true);
    DenseTensor(const std::string& name, const std::string& indices);

    static uint64_t getSize(int ndim, const std::vector<int>& len, const std::vector<int>& ld);

    void print() const;

    void mult(const T alpha, const DenseTensor<T>& A, const std::string& idx_A,
                             const DenseTensor<T>& B, const std::string& idx_B,
              const T beta,                           const std::string& idx_C);

    void sum(const T alpha, const DenseTensor<T>& A, const std::string& idx_A,
             const T beta,                           const std::string& idx_B);

    void scale(const T alpha, const std::string& idx_A);

    //DenseTensor<T> slice(const std::vector<int>& start, const std::vector<int>& len);

};

}

}

#endif
