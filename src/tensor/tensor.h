/*
 * Copyright (C) 2014  Justin Turney
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

#if !defined(AMBIT_LIB_TENSOR_TENSOR)
#define AMBIT_LIB_TENSOR_TENSOR

#include "exception.h"
#include "indices.h"
#include <string>

#if defined(DEBUG)
#include <iostream>
#endif

namespace ambit { namespace tensor {

// forward declarations
template <typename derived, typename T>
struct tensor;

template <typename Derived, typename T>
struct indexed_tensor;

template <typename derived, typename T>
struct scaled_tensor;

template <typename derived, typename T>
struct inverted_tensor;

template <typename derived, typename T>
struct indexed_tensor_multiplication;

template <typename derived, typename T>
struct tensor_division;

template <typename T>
struct keyvalue_pair
{
    int64_t key;
    T value;

    keyvalue_pair(int64_t k, T v) : key(k), value(v) {}
};

struct key_generator1
{
    const index_range& range1;

    key_generator1(const index_range& r1) : range1(r1) {}

    int64_t operator()(int64_t p) const { return p; }
};

struct key_generator2
{
    const index_range& range1;
    const index_range& range2;

    key_generator2(const index_range& r1, const index_range& r2) : range1(r1), range2(r2) {}

    int64_t operator()(int64_t p, int64_t q) const { return p * range2.length() + q; }
};

template <typename Derived, typename T>
struct tensor_base
{
    // this is the POD-type of the tensor.
    typedef T data_type;

    tensor_base(const std::string& name, int ndim=0) : name_(name), ndim_(ndim) {};
    virtual ~tensor_base() {}

    const std::string& name() const { return name_; }

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    int dimension() const { return ndim_; }

    // operators with scalar
    Derived& operator=(const T val)
    {
        sum(val, (T)0);
        return derived();
    }

    Derived& operator+=(const T val)
    {
        sum(val, (T)1);
        return derived();
    }

    Derived& operator*=(const T val)
    {
        multiply(val);
        return derived();
    }

    Derived& operator/=(const T val)
    {
        multiply(1.0/val);
        return derived();
    }

    /*****************************************************************************************************************
     *
     * STUBS
     *
     *****************************************************************************************************************/

    // this = alpha * this + beta * A * B
    virtual void multiply(const T& alpha, const indexed_tensor<Derived, T>& A,
                                          const indexed_tensor<Derived, T>& B,
                          const T& beta, const std::string& index_C) = 0;

    // this = alpha * A
    virtual void sum(const T& alpha, const indexed_tensor<Derived, T>& A,
                     const T& beta,  const std::string& index_B) = 0;

    // this = alpha * this
    //virtual void scale(const T& alpha) = 0;

protected:
    std::string name_;
    int ndim_;
};

template <typename Derived, typename T>
struct indexed_tensor
{
    typedef T data_type;

    Derived& tensor_;
    std::string index_;
    data_type factor_;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    indexed_tensor(Derived& tensor, const std::string& index, const T& factor=(T)1)
        : tensor_(tensor), index_(index), factor_(factor)
    {
        std::vector<std::string> indices = split_indices(index_);
        if (indices.size() != tensor.dimension())
            throw invalid_ndim_error();

        // ensure the indices given are valid. this throws if invalid.
        // TODO: We may not want to do this. When using implicit() this will likely throw.
//        index_range::find(indices);
    }

    // unary negation
    indexed_tensor<Derived,T> operator-() const
    {
        indexed_tensor<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    // multiplication expression
    indexed_tensor_multiplication<Derived, T> operator*(const indexed_tensor<Derived, T>& other) const
    {
        return indexed_tensor_multiplication<Derived, T>(*(this), other);
    }

    /*****************************************************************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *****************************************************************************************************************/
    indexed_tensor<Derived, T>& operator=(const indexed_tensor<Derived, T>& other)
    {
        tensor_.sum(other.factor_, other,
                    (T)0,          index_);
        return *this;
    }

    /*****************************************************************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *****************************************************************************************************************/
    indexed_tensor<Derived, T>& operator=(const indexed_tensor_multiplication<Derived,T>& other)
    {
        tensor_.multiply(other.factor_, other.A_,
                                        other.B_,
                         (T)0,          index_);
        return *this;
    }

    indexed_tensor<Derived, T>& operator+=(const indexed_tensor_multiplication<Derived,T>& other)
    {
        tensor_.multiply(other.factor_, other.A_,
                                        other.B_,
                         factor_,       index_);
        return *this;
    }

    indexed_tensor<Derived, T>& operator-=(const indexed_tensor_multiplication<Derived,T>& other)
    {
        tensor_.multiply(-other.factor_, other.A_,
                                         other.B_,
                         factor_,        index_);
        return *this;
    }
};

template <typename Derived, typename T>
struct indexable_tensor : public tensor_base<Derived, T>
{
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    indexable_tensor(const std::string& name, int ndim=0) : tensor_base<Derived, T>(name, ndim) {}
    virtual ~indexable_tensor() {}

    std::string implicit() const
    {
        // TODO: make this a comma separated string
        std::string idxs(tensor_base<Derived, T>::dimension(), ' ');
        for (int i=0; i<tensor_base<Derived, T>::dimension(); ++i)
            idxs[i] = (char)('A'+i);
        return idxs;
    }

    // explicit indexing operations
    indexed_tensor<Derived, T> operator[](const std::string& idx)
    {
        return indexed_tensor<Derived, T>(derived(), idx);
    }

    indexed_tensor<const Derived, T> operator[](const std::string& idx) const
    {
        return indexed_tensor<const Derived, T>(derived(), idx);
    }

    //Derived& operator=(const Derived& other)
    //{
    //    const std::string index = implicit();
    //    indexed_tensor<Derived, T> this_(derived(), index);
    //    indexed_tensor<Derived, T> other_(other, index);
    //    this_ = other_;
    //    return derived();
    //}

protected:

    void check_indices(const std::string& idx)
    {
        int dim = 0;
        // check if there is a comma in the index list
        if (idx.find(",") != std::string::npos) {
            // found a comma, use our index_range functions to check the indices
            dim = split_indices(idx).size();
        }
        else {
            dim = idx.size();
        }

        if (dim != tensor_base<Derived, T>::dimension())
            throw invalid_ndim_error();
    }
};

template <typename Derived, typename T>
struct indexed_tensor_multiplication
{
    const indexed_tensor_multiplication& operator=(const indexed_tensor_multiplication<Derived, T>& other) = delete;
    //indexed_tensor_multiplication(const indexed_tensor_multiplication<Derived, T>& other) = delete;

    const indexed_tensor<Derived, T>& A_;
    const indexed_tensor<Derived, T>& B_;
    T factor_;

    //template <typename Derived1, typename Derived2>
    indexed_tensor_multiplication(const indexed_tensor<Derived, T>& A, const indexed_tensor<Derived, T>& B)
        : A_(A), B_(B), factor_(A.factor_ * B_.factor_) {}
};

}}

#endif
