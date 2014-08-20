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
template <typename Derived, typename T>
struct indexed_tensor;

template <typename derived, typename T>
struct scaled_tensor;

template <typename derived, typename T>
struct inverted_tensor;

template <typename derived, typename T>
struct indexed_tensor_multiplication;

template <typename derived, typename T>
struct indexed_tensor_subtraction;

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

struct key_generator3
{
    const index_range& range1;
    const index_range& range2;
    const index_range& range3;

    key_generator3(const index_range& r1,
                   const index_range& r2,
                   const index_range& r3)
        : range1(r1), range2(r2), range3(r3) {}

    int64_t operator()(int64_t p, int64_t q, int64_t r) const {
        return ((p * range2.length()) + range3.length() * q) + r;
    }
};

struct key_generator4
{
    const index_range& range1;
    const index_range& range2;
    const index_range& range3;
    const index_range& range4;

    key_generator4(const index_range& r1,
                   const index_range& r2,
                   const index_range& r3,
                   const index_range& r4)
        : range1(r1), range2(r2), range3(r3), range4(r4) {}

    int64_t operator()(int64_t p, int64_t q, int64_t r, int64_t s) const {
        int64_t v = (((p) * range2.length() + q) * range3.length() + r) * range4.length() + s;
        return v;
    }
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

    // this(C) = alpha * A + beta * B
    virtual void sum(const T& alpha, const indexed_tensor<Derived, T>& A,
                     const T& beta,  const indexed_tensor<Derived, T>& B,
                     const std::string& index_C) = 0;

    virtual T dot(const indexed_tensor<Derived, T>& A,
                  const std::string& index_B) = 0;

    // this = alpha * this
    //virtual void scale(const T& alpha) = 0;

protected:

    void set_dimension(int ndim) { ndim_ = ndim; }

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
        if (!index_.empty() && indices.size() != tensor.dimension())
            throw invalid_ndim_error();

        // ensure the indices given are valid. this throws if invalid.
        // TODO: We may not want to do this. When using implicit() this will likely throw.
//        index_range::find(indices);
    }

    // dot product
    T dot(const indexed_tensor<Derived, T>& other) const
    {
        return tensor_.dot(other, index_);
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

    // subtraction expression
    indexed_tensor_subtraction<Derived, T> operator-(const indexed_tensor<Derived, T>& other) const
    {
        return indexed_tensor_subtraction<Derived, T>(*(this), other);
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
     * Binary tensor operations
     *
     *****************************************************************************************************************/
    indexed_tensor<Derived, T>& operator=(const indexed_tensor_multiplication<Derived,T>& other)
    {
        tensor_.multiply(other.factor_, other.A_,
                                        other.B_,
                         (T)0,          index_);
        return *this;
    }

    indexed_tensor<Derived, T>& operator=(const indexed_tensor_subtraction<Derived,T>& other)
    {
        tensor_.sum(other.A_.factor_, other.A_,
                    -other.B_.factor_, other.B_,
                    index_);
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

    indexed_tensor<Derived, T>& operator+=(const indexed_tensor<Derived, T>& other)
    {
        tensor_.sum(1.0, other, 1.0, index_);
        return *this;
    }

    indexed_tensor<Derived, T>& operator-=(const indexed_tensor<Derived, T>& other)
    {
        tensor_.sum(-1.0, other, 1.0, index_);
        return *this;
    }
};

template <typename Derived, typename T>
struct indexable_tensor : public tensor_base<Derived, T>
{
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    indexable_tensor(const std::string& name, const std::string& indices, int ndim=0) : tensor_base<Derived, T>(name, ndim), indices_(indices)
    {
        ir_ = index_range::find(split_indices(indices));
        tensor_base<Derived, T>::set_dimension(ir_.size());
        check_indices(indices);
    }

    virtual ~indexable_tensor() {}

    std::string implicit() const
    {
        // TODO: make this a comma separated string
        std::string idxs(tensor_base<Derived, T>::dimension(), ' ');
        for (int i=0; i<tensor_base<Derived, T>::dimension(); ++i)
            idxs[i] = (char)('A'+i);
        return idxs;
    }

    const std::vector<index_range>& index_ranges() const { return ir_; }

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

    const std::string& indices_;
    std::vector<index_range> ir_;
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

template <typename Derived, typename T>
struct indexed_tensor_subtraction
{
    const indexed_tensor_subtraction& operator=(const indexed_tensor_subtraction<Derived, T>& other) = delete;

    const indexed_tensor<Derived, T>& A_;
    const indexed_tensor<Derived, T>& B_;

    indexed_tensor_subtraction(const indexed_tensor<Derived, T>& A, const indexed_tensor<Derived, T>& B)
        : A_(A), B_(B) {}
};

}}

#if defined(HAVE_MPI)
#include "cyclops/tensor.h"
#endif

#endif
