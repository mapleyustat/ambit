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

#if !defined(AMBIT_LIB_TENSOR_INDEXABLE_TENSOR)
#define AMBIT_LIB_TENSOR_INDEXABLE_TENSOR

#include "tensor.h"

namespace ambit { namespace tensor {

template <typename derived, typename T> struct indexable_tensor;
template <typename derived, typename T> struct indexed_tensor;
template <typename derived, typename T> struct indexed_tensor_mult;

#define INHERIT_FROM_INDEXABLE_TENSOR(derived,T) \
    protected: \
        using ambit::tensor::indexable_tensor< derived, T >::ndim; \
    public: \
        using ambit::tensor::indexable_tensor< derived, T >::mult; \
        using ambit::tensor::indexable_tensor< derived, T >::sum; \
        using ambit::tensor::indexable_tensor< derived, T >::scale; \
        using ambit::tensor::indexable_tensor< derived, T >::dot; \
        using ambit::tensor::indexable_tensor< derived, T >::operator=; \
        using ambit::tensor::indexable_tensor< derived, T >::operator+=; \
        using ambit::tensor::indexable_tensor< derived, T >::operator-=; \
        using ambit::tensor::indexable_tensor< derived, T >::operator[]; \
        using ambit::tensor::tensor< derived,T >::get_derived; \
        using ambit::tensor::tensor< derived,T >::operator*=; \
        using ambit::tensor::tensor< derived,T >::operator/=; \
        using ambit::tensor::tensor< derived,T >::operator*; \
        using ambit::tensor::tensor< derived,T >::operator/; \
        using ambit::tensor::tensor< derived,T >::get_name; \
        derived & operator=(const derived & other) \
        { \
            sum((T)1, other, (T)0); \
            return *this; \
        } \
    private:

template <typename derived, typename T>
struct indexable_tensor_base
{
protected:
    int ndim;

public:
    indexable_tensor_base(const int ndim=0) : ndim(ndim) {}

    virtual ~indexable_tensor_base() {}

    derived& get_derived() { return static_cast<derived&>(*this); }
    const derived& get_derived() const { return static_cast<const derived&>(*this); }

    int get_dimension() const { return ndim; }

    std::string implicit() const
    {
        std::string inds(ndim, ' ');
        for (int i=0; i<ndim; ++i) inds[i] = (char)('A'+i);
        return inds;
    }

    /**********************************************************************
     *
     * Explicit indexing operations
     *
     *********************************************************************/
    indexed_tensor<derived,T> operator[](const std::string& idx)
    {
        return indexed_tensor<derived,T>(get_derived(), idx);
    }

    indexed_tensor<const derived,T> operator[](const std::string& idx) const
    {
        return indexed_tensor<const derived,T>(get_derived(), idx);
    }

    /**********************************************************************
     *
     * Implicitly indexed binary operations (inner product, trace, and weighting)
     *
     *********************************************************************/
    template <typename cvderived>
    derived& operator=(const indexed_tensor_mult<cvderived,T>& other)
    {
        (*this)[implicit()] = other;
        return get_derived();
    }

    template <typename cvderived>
    derived& operator+=(const indexed_tensor_mult<cvderived,T>& other)
    {
        (*this)[implicit()] += other;
        return get_derived();
    }

    template <typename cvderived>
    derived& operator-=(const indexed_tensor_mult<cvderived,T>& other)
    {
        (*this)[implicit()] -= other;
        return get_derived();
    }

    /**********************************************************************
     *
     * Implicitly indexed unary operations (assignment and summation)
     *
     *********************************************************************/
    template <typename cvderived>
    derived& operator=(const indexed_tensor<cvderived,T>& other)
    {
        (*this)[implicit()] = other;
        return get_derived();
    }

    template <typename cvderived>
    derived& operator+=(const indexed_tensor<cvderived,T>& other)
    {
        (*this)[implicit()] += other;
        return get_derived();
    }

    template <typename cvderived>
    derived& operator-=(const indexed_tensor<cvderived,T>& other)
    {
        (*this)[implicit()] -= other;
        return get_derived();
    }

    /**********************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *********************************************************************/
    virtual void mult(const T alpha, const derived& A, const std::string& idx_A,
                                     const derived& B, const std::string& idx_B,
                      const T beta,                    const std::string& idx_C) = 0;


    /**********************************************************************
     *
     * Unary tensor operations (summation)
     *
     *********************************************************************/
    virtual void sum(const T alpha, const derived& A, const std::string& idx_A,
                     const T beta,                    const std::string& idx_B) = 0;


    /**********************************************************************
     *
     * Scalar operations
     *
     *********************************************************************/
    virtual void scale(const T alpha, const std::string& idx_A) = 0;

    virtual T dot(const derived& A, const std::string& idx_A,
                                    const std::string& idx_B) const = 0;
};

template <typename derived, typename T>
struct indexable_tensor : public indexable_tensor_base<derived, T>, public tensor<derived, T>
{
    INHERIT_FROM_TENSOR(derived, T)

protected:
    using indexable_tensor_base<derived, T>::ndim;

public:
    using indexable_tensor_base<derived,T>::scale;
    using indexable_tensor_base<derived,T>::dot;
    using indexable_tensor_base<derived,T>::mult;
    using indexable_tensor_base<derived,T>::sum;
    using indexable_tensor_base<derived,T>::implicit;

    indexable_tensor(const std::string& name, const int ndim = 0)
        : indexable_tensor_base<derived, T>(ndim), tensor<derived,T>(name) {}
    virtual ~indexable_tensor() {}

    /**********************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *********************************************************************/
    void mult(const T alpha)
    {
        scale(alpha);
    }

    void mult(const T alpha, const derived& A,
                             const derived& B,
              const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.get_dimension() || ndim != B_.get_dimension()) throw invalid_ndim_error();
#endif //VALIDATE_INPUTS

        mult(alpha, A, A.implicit(),
                    B, B.implicit(),
              beta,      implicit());
    }

    /**********************************************************************
     *
     * Unary tensor operations (summation)
     *
     *********************************************************************/
    void sum(const T alpha, const T beta)
    {
        derived tensor("alpha", get_derived(), alpha);
        beta*(*this)[implicit()] = tensor[""];
    }

    void sum(const T alpha, const derived& A, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.get_dimension()) throw invalid_ndim_error();
#endif //VALIDATE_INPUTS

        sum(alpha, A, A.implicit(),
             beta,      implicit());
    }

    /**********************************************************************
     *
     * Scalar operations
     *
     *********************************************************************/
    void scale(const T alpha)
    {
        scale(alpha, implicit());
    }

    T dot(const derived& A) const
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.get_dimension()) throw invalid_ndim_error();
#endif //VALIDATE_INPUTS

        return dot(A, A.implicit(),
                        implicit());
    }
};

template <typename derived, typename T>
struct indexed_tensor
{
    derived& tensor_;
    std::string idx_;
    T factor_;

    template <typename cvderived>
    indexed_tensor(const indexed_tensor<cvderived,T>& other)
    : tensor_(other.tensor_), idx_(other.idx_), factor_(other.factor_) {}

    indexed_tensor(derived& tensor, const std::string& idx, const T factor=(T)1)
    : tensor_(tensor), idx_(idx), factor_(factor)
    {
        if (idx.size() != tensor.get_dimension()) throw invalid_ndim_error();
    }

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    indexed_tensor<derived,T> operator-() const
    {
        indexed_tensor<derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Unary tensor operations (summation)
     *
     *********************************************************************/
    indexed_tensor<derived,T>& operator=(const indexed_tensor<derived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, (T)0, idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor<derived,T>& operator=(const indexed_tensor<cvderived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, (T)0, idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor<derived,T>& operator+=(const indexed_tensor<cvderived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, factor_, idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor<derived,T>& operator-=(const indexed_tensor<cvderived,T>& other)
    {
        tensor_.sum(-other.factor_, other.tensor_, other.idx_, factor_, idx_);
        return *this;
    }

    /**********************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *********************************************************************/
    template <typename cvderived>
    indexed_tensor<derived,T>& operator=(const indexed_tensor_mult<cvderived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.A_.idx_,
                                    other.B_.tensor_, other.B_.idx_,
                              (T)0,                            idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor<derived,T>& operator+=(const indexed_tensor_mult<cvderived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.A_.idx_,
                                    other.B_.tensor_, other.B_.idx_,
                           factor_,                            idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor<derived,T>& operator-=(const indexed_tensor_mult<cvderived,T>& other)
    {
        tensor_.mult(-other.factor_, other.A_.tensor_, other.A_.idx_,
                                     other.B_.tensor_, other.B_.idx_,
                            factor_,                            idx_);
        return *this;
    }

    template <typename cvderived>
    indexed_tensor_mult<derived,T> operator*(const indexed_tensor<cvderived,T>& other) const
    {
        return indexed_tensor_mult<derived,T>(*this, other);
    }

    template <typename cvderived>
    indexed_tensor_mult<derived,T> operator*(const ScaledTensor<cvderived,T>& other) const
    {
        cvderived& B = other.tensor_.get_derived();

        return indexed_tensor_mult<derived,T>(*this, B[B.implicit()]*other.factor_);
    }

    template <typename cvderived>
    indexed_tensor_mult<derived,T> operator*(const indexable_tensor<cvderived,T>& other) const
    {
        return indexed_tensor_mult<derived,T>(*this, other[other.implicit()]);
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    indexed_tensor<derived,T> operator*(const T factor) const
    {
        indexed_tensor<derived,T> it(*this);
        it.factor_ *= factor;
        return it;
    }

    friend indexed_tensor<derived,T> operator*(const T factor, const indexed_tensor<derived,T>& other)
    {
        return other*factor;
    }

    indexed_tensor<derived,T>& operator*=(const T factor)
    {
        tensor_.scale(factor, idx_);
        return *this;
    }

    indexed_tensor<derived,T>& operator=(const T val)
    {
        derived tensor(tensor_, val);
        *this = tensor[""];
        return *this;
    }

    indexed_tensor<derived,T>& operator+=(const T val)
    {
        derived tensor(tensor_, val);
        *this += tensor[""];
        return *this;
    }

    indexed_tensor<derived,T>& operator-=(const T val)
    {
        derived tensor(tensor_, val);
        *this -= tensor[""];
        return *this;
    }
};

template <class derived1, class derived2, class T>
//typename std::enable_if<std::is_same<const derived1, const derived2>::value,IndexedTensorMult<derived1,T> >::type
indexed_tensor_mult<derived1,T>
operator*(const indexable_tensor_base<derived1,T>& t1, const indexed_tensor<derived2,T>& t2)
{
    return indexed_tensor_mult<derived1,T>(t1[t1.implicit()], t2);
}

template <class derived1, class derived2, class T>
//typename std::enable_if<std::is_same<const derived1, const derived2>::value,IndexedTensorMult<derived1,T> >::type
indexed_tensor_mult<derived1,T>
operator*(const ScaledTensor<derived1,T>& t1, const indexed_tensor<derived2,T>& t2)
{
    derived1& A = t1.tensor_.get_derived();

    return indexed_tensor_mult<derived1,T>(A[A.implicit()]*t1.factor_, t2);
}

template <typename derived, typename T>
struct indexed_tensor_mult
{
private:
    const indexed_tensor_mult& operator=(const indexed_tensor_mult<derived, T>& other);

public:
    indexed_tensor<const derived,T> A_;
    indexed_tensor<const derived,T> B_;
    T factor_;

    template <class derived1, class derived2>
    indexed_tensor_mult(const indexed_tensor<derived1,T>& A, const indexed_tensor<derived2,T>& B)
        : A_(A), B_(B), factor_(A.factor_*B.factor_) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    indexed_tensor_mult<derived,T> operator-() const
    {
        indexed_tensor_mult<derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    indexed_tensor_mult<derived,T> operator*(const T factor) const
    {
        indexed_tensor_mult<derived,T> ret(*this);
        ret.factor_ *= factor;
        return ret;
    }

    indexed_tensor_mult<derived,T> operator/(const T factor) const
    {
        indexed_tensor_mult<derived,T> ret(*this);
        ret.factor_ /= factor;
        return ret;
    }

    friend indexed_tensor_mult<derived,T> operator*(const T factor, const indexed_tensor_mult<derived,T>& other)
    {
        return other*factor;
    }
};

} // namespace tensor

/**************************************************************************
 *
 * Tensor to scalar operations
 *
 *************************************************************************/
template <class derived, typename T>
T scalar(const tensor::indexed_tensor_mult<derived,T>& itm)
{
    return itm.factor_*itm.B_.tensor_.dot(itm.A_.tensor_, itm.A_.idx_,
                                                          itm.B_.idx_);
}

} // namespace ambit

#endif
