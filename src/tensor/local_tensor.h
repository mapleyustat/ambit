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

#if !defined(AMBIT_LIB_TENSOR_LOCAL_TENSOR)
#define AMBIT_LIB_TENSOR_LOCAL_TENSOR

#include "indexable_tensor.h"
#include "indices.h"
#include <util/memory.h>

#include <cassert>
#include <cfloat>
#include <vector>

#ifdef DEBUG
#include <iostream>
#include <util/prettyprint.h>
#endif

namespace ambit {

namespace tensor {

#define INHERIT_FROM_LOCAL_TENSOR(Derived, T) \
    protected: \
        using ambit::tensor::LocalTensor< Derived, T >::len; \
        using ambit::tensor::LocalTensor< Derived, T >::ld; \
        using ambit::tensor::LocalTensor< Derived, T >::size; \
        using ambit::tensor::LocalTensor< Derived, T >::data; \
    public: \
        using ambit::tensor::LocalTensor< Derived, T >::CLONE; \
        using ambit::tensor::LocalTensor< Derived, T >::REFERENCE; \
        using ambit::tensor::LocalTensor< Derived, T >::REPLACE; \
        using ambit::tensor::LocalTensor< Derived, T >::getSize; \
    INHERIT_FROM_INDEXABLE_TENSOR(Derived, T) \
    friend class ambit::tensor::LocalTensor< Derived, T >;

#define CHECK_RETURN_VALUE(ret) \
    switch (ret) \
    { \
        case kTensorReturnCodeOutOfBounds: \
            throw OutOfBoundsError(); \
            break; \
        case kTensorReturnCodeLengthMismatch: \
            throw LengthMismatchError(); \
            break; \
        case kTensorReturnCodeIndexMismatch: \
            throw IndexMismatchError(); \
            break; \
        case kTensorReturnCodeInvalidNDim: \
            throw InvalidNdimError(); \
            break; \
        case kTensorReturnCodeInvalidLength: \
            throw InvalidLengthError(); \
            break; \
        case kTensorReturnCodeInvalidLD: \
            throw InvalidLdError(); \
            break; \
        case kTensorReturnCodeLDTooSmall: \
            throw LdTooSmallError(); \
            break; \
        case kTensorReturnCodeSymmetryMismatch: \
            throw SymmetryMismatchError(); \
            break; \
        case kTensorReturnCodeInvalidSymmetry: \
            throw InvalidSymmetryError(); \
            break; \
        case kTensorReturnCodeInvalidStart: \
            throw InvalidStartError(); \
            break; \
        default: \
            break; \
    }

#define VALIDATE_TENSOR_THROW(ndim,len,ld,sym) CHECK_RETURN_VALUE(validate_tensor(ndim,len,ld,sym))

template <class Derived, typename T>
struct LocalTensor : public IndexableTensor< Derived, T>
{
    INHERIT_FROM_INDEXABLE_TENSOR(Derived,T)

protected:
    std::vector<int> len;
    std::vector<int> ld;
    uint64_t size;
    T* data;
    bool isAlloced;

public:
    enum CopyType { CLONE, REFERENCE, REPLACE };

    LocalTensor(const std::string& name, const T& val = (T)0)
        : IndexableTensor<Derived,T>(name, 0), len(0), ld(0), size(1)
    {
        data = SAFE_MALLOC(T, 1);
        isAlloced = true;
        data[0] = val;
    }

    LocalTensor(const Derived& A)
        : IndexableTensor<Derived,T>(A.name, A.ndim), len(A.len), ld(A.ld), size(A.size)
    {
        data = SAFE_MALLOC(T, size);
        std::copy(A.data, A.data+size, data);
        isAlloced = true;
    }

    LocalTensor(const std::string& name, const Derived& A)
        : IndexableTensor<Derived,T>(name, A.ndim), len(A.len), ld(A.ld), size(A.size)
    {
        data = SAFE_MALLOC(T, size);
        std::copy(A.data, A.data+size, data);
        isAlloced = true;
    }

    LocalTensor(const std::string& name, Derived& A, const CopyType type=CLONE)
        : IndexableTensor<Derived,T>(name, A.ndim), len(A.len), ld(A.ld), size(A.size)
    {
        switch(type)
        {
            case CLONE:
                data = SAFE_MALLOC(T, size);
                std::copy(A.data, A.data+size, data);
                isAlloced = true;
                break;
            case REFERENCE:
                data = A.data;
                isAlloced = false;
                break;
            case REPLACE:
                data = A.data;
                isAlloced = A.isAlloced;
                A.isAlloced = false;
                break;
        }
    }

    LocalTensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld_, uint64_t size, T* data_, bool zero=false)
        : IndexableTensor<Derived,T>(name, len.size()), len(len), ld(ld_), size(size)
    {
        size_t ndim = len.size();
        if (ld.size() != ndim) {
            ld.resize(ndim);
            ld[0] = 1;
            for (int i=1; i<ndim; ++i)
                ld[i] = ld[i-1]*len[i-1];
        }

#ifdef VALIDATE_INPUTS
        if (validate_tensor(ndim, len.data(), ld.data(), NULL) != TENSOR_SUCCESS)
            throw std::runtime_error("not a valid tensor");
#endif

        data = data_;
        isAlloced = false;
        if (zero)
            std::fill(data, data+size, (T)0);
    }

    LocalTensor(const std::string& name, const std::string& indices)
        : IndexableTensor<Derived,T>(name, indices.size())
    {
        // Make sure the indices are known.
        std::vector<IndexRange> ind = IndexRange::find(split_indices(indices));

        // Check rank of the indices
#ifdef DEBUG
        std::cout << ind << std::endl;
#endif
        for (auto& i : ind) {
            if (i.start.size() > 1)
                throw InvalidLengthError();
        }
        isAlloced = false;
    }

    LocalTensor(const std::string& name, const std::vector<int>& len, const std::vector<int>& ld_, uint64_t size_, bool zero=true)
        : IndexableTensor<Derived,T>(name, len.size()), len(len), ld(ld_), size(size_)
    {
        size_t ndim = len.size();

        if (ld.size() != ndim) {
            ld.resize(ndim);
            ld[0] = 1;
            for (int i=1; i<ndim; ++i)
                ld[i] = ld[i-1]*len[i-1];
        }

#ifdef VALIDATE_INPUTS
        if (validate_tensor(ndim, len.data(), ld.data(), NULL) != TENSOR_SUCCESS)
            throw std::runtime_error("not a valid tensor");
#endif

        data = SAFE_MALLOC(T, size);
        isAlloced = true;
        if (zero)
            std::fill(data, data+size, (T)0);
    }

    ~LocalTensor()
    {
        if (isAlloced)
            FREE(data);
    }

    const std::vector<int>& getLengths() const { return len; }
    const std::vector<int>& getLeadingDims() const { return ld; }
    uint64_t getSize() const { return size; }

    void div(const T alpha, const Derived& A,
                            const Derived& B, const T beta)
    {
        assert(size == A.size && size == B.size);

        for (uint64_t i=0; i<size; ++i) {
            if (std::abs(B.data[i]) > DBL_MIN)
                data[i] = beta*data[i] + alpha*A.data[i]/B.data[i];
        }
    }

    void invert(const T alpha, const Derived& A, const T beta)
    {
        assert(size == A.size);

        for (uint64_t i=0; i<size; ++i) {
            if (std::abs(A.data[i]) > DBL_MIN)
                data[i] = beta*data[i] + alpha/A.data[i];
        }
    }

    virtual void print() const = 0;

    T* getData() { return data; }
    const T* getData() const { return data; }

    T dot(const Derived& A, const std::string& idx_A,
                            const std::string& idx_B) const
    {
        Derived dt("scalar");
        dt.mult(1, A,            idx_A,
                   getDerived(), idx_B,
                0,               "");
        return dt.getData()[0];
    }
};

}

}

#endif
