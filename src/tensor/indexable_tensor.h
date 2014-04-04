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

template <typename Derived, typename T> struct indexable_tensor;
template <typename Derived, typename T> struct indexed_tensor;
template <typename Derived, typename T> struct indexed_tensor_mult;

#define INHERIT_FROM_INDEXABLE_TENSOR(Derived,T) \
    protected: \
        using ambit::tensor::indexable_tensor< Derived, T >::ndim; \
    public: \
        using ambit::tensor::indexable_tensor< Derived, T >::mult; \
        using ambit::tensor::indexable_tensor< Derived, T >::sum; \
        using ambit::tensor::indexable_tensor< Derived, T >::scale; \
        using ambit::tensor::indexable_tensor< Derived, T >::dot; \
        using ambit::tensor::indexable_tensor< Derived, T >::operator=; \
        using ambit::tensor::indexable_tensor< Derived, T >::operator+=; \
        using ambit::tensor::indexable_tensor< Derived, T >::operator-=; \
        using ambit::tensor::indexable_tensor< Derived, T >::operator[]; \
        using ambit::tensor::tensor< Derived,T >::get_derived; \
        using ambit::tensor::tensor< Derived,T >::operator*=; \
        using ambit::tensor::tensor< Derived,T >::operator/=; \
        using ambit::tensor::tensor< Derived,T >::operator*; \
        using ambit::tensor::tensor< Derived,T >::operator/; \
        using ambit::tensor::tensor< Derived,T >::get_name; \
        Derived & operator=(const Derived & other) \
        { \
            sum((T)1, other, (T)0); \
            return *this; \
        } \
    private:

template <typename Derived, typename T>
struct indexable_tensor_base
{
protected:
    int ndim;

public:
    indexable_tensor_base(const int ndim=0) : ndim(ndim) {}

    virtual ~indexable_tensor_base() {}

    Derived& get_derived() { return static_cast<Derived&>(*this); }
    const Derived& get_derived() const { return static_cast<const Derived&>(*this); }

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
    indexed_tensor<Derived,T> operator[](const std::string& idx)
    {
        return indexed_tensor<Derived,T>(get_derived(), idx);
    }

    indexed_tensor<const Derived,T> operator[](const std::string& idx) const
    {
        return indexed_tensor<const Derived,T>(get_derived(), idx);
    }

    /**********************************************************************
     *
     * Implicitly indexed binary operations (inner product, trace, and weighting)
     *
     *********************************************************************/
    template <typename cvDerived>
    Derived& operator=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        (*this)[implicit()] = other;
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator+=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        (*this)[implicit()] += other;
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator-=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        (*this)[implicit()] -= other;
        return get_derived();
    }

    /**********************************************************************
     *
     * Implicitly indexed unary operations (assignment and summation)
     *
     *********************************************************************/
    template <typename cvDerived>
    Derived& operator=(const indexed_tensor<cvDerived,T>& other)
    {
        (*this)[implicit()] = other;
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator+=(const indexed_tensor<cvDerived,T>& other)
    {
        (*this)[implicit()] += other;
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator-=(const indexed_tensor<cvDerived,T>& other)
    {
        (*this)[implicit()] -= other;
        return get_derived();
    }

    /**********************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *********************************************************************/
    virtual void mult(const T alpha, const Derived& A, const std::string& idx_A,
                                     const Derived& B, const std::string& idx_B,
                      const T beta,                    const std::string& idx_C) = 0;


    /**********************************************************************
     *
     * Unary tensor operations (summation)
     *
     *********************************************************************/
    virtual void sum(const T alpha, const Derived& A, const std::string& idx_A,
                     const T beta,                    const std::string& idx_B) = 0;


    /**********************************************************************
     *
     * Scalar operations
     *
     *********************************************************************/
    virtual void scale(const T alpha, const std::string& idx_A) = 0;

    virtual T dot(const Derived& A, const std::string& idx_A,
                                    const std::string& idx_B) const = 0;
};

template <typename Derived, typename T>
struct indexable_tensor : public indexable_tensor_base<Derived, T>, public tensor<Derived, T>
{
    INHERIT_FROM_TENSOR(Derived, T)

protected:
    using indexable_tensor_base<Derived, T>::ndim;

public:
    using indexable_tensor_base<Derived,T>::scale;
    using indexable_tensor_base<Derived,T>::dot;
    using indexable_tensor_base<Derived,T>::mult;
    using indexable_tensor_base<Derived,T>::sum;
    using indexable_tensor_base<Derived,T>::implicit;

    indexable_tensor(const std::string& name, const int ndim = 0)
        : indexable_tensor_base<Derived, T>(ndim), tensor<Derived,T>(name) {}
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

    void mult(const T alpha, const Derived& A,
                             const Derived& B,
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
        Derived tensor("alpha", get_derived(), alpha);
        beta*(*this)[implicit()] = tensor[""];
    }

    void sum(const T alpha, const Derived& A, const T beta)
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

    T dot(const Derived& A) const
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.get_dimension()) throw invalid_ndim_error();
#endif //VALIDATE_INPUTS

        return dot(A, A.implicit(),
                        implicit());
    }
};

template <typename Derived, typename T>
struct indexed_tensor
{
    Derived& tensor_;
    std::string idx_;
    T factor_;

    template <typename cvDerived>
    indexed_tensor(const indexed_tensor<cvDerived,T>& other)
    : tensor_(other.tensor_), idx_(other.idx_), factor_(other.factor_) {}

    indexed_tensor(Derived& tensor, const std::string& idx, const T factor=(T)1)
    : tensor_(tensor), idx_(idx), factor_(factor)
    {
        if (idx.size() != tensor.get_dimension()) throw InvalidNdimError();
    }

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    indexed_tensor<Derived,T> operator-() const
    {
        indexed_tensor<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Unary tensor operations (summation)
     *
     *********************************************************************/
    indexed_tensor<Derived,T>& operator=(const indexed_tensor<Derived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, (T)0, idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator=(const indexed_tensor<cvDerived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, (T)0, idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator+=(const indexed_tensor<cvDerived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, other.idx_, factor_, idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator-=(const indexed_tensor<cvDerived,T>& other)
    {
        tensor_.sum(-other.factor_, other.tensor_, other.idx_, factor_, idx_);
        return *this;
    }

    /**********************************************************************
     *
     * Binary tensor operations (multiplication)
     *
     *********************************************************************/
    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.A_.idx_,
                                    other.B_.tensor_, other.B_.idx_,
                              (T)0,                            idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator+=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.A_.idx_,
                                    other.B_.tensor_, other.B_.idx_,
                           factor_,                            idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor<Derived,T>& operator-=(const indexed_tensor_mult<cvDerived,T>& other)
    {
        tensor_.mult(-other.factor_, other.A_.tensor_, other.A_.idx_,
                                     other.B_.tensor_, other.B_.idx_,
                            factor_,                            idx_);
        return *this;
    }

    template <typename cvDerived>
    indexed_tensor_mult<Derived,T> operator*(const indexed_tensor<cvDerived,T>& other) const
    {
        return indexed_tensor_mult<Derived,T>(*this, other);
    }

    template <typename cvDerived>
    indexed_tensor_mult<Derived,T> operator*(const ScaledTensor<cvDerived,T>& other) const
    {
        cvDerived& B = other.tensor_.get_derived();

        return indexed_tensor_mult<Derived,T>(*this, B[B.implicit()]*other.factor_);
    }

    template <typename cvDerived>
    indexed_tensor_mult<Derived,T> operator*(const indexable_tensor<cvDerived,T>& other) const
    {
        return indexed_tensor_mult<Derived,T>(*this, other[other.implicit()]);
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    indexed_tensor<Derived,T> operator*(const T factor) const
    {
        indexed_tensor<Derived,T> it(*this);
        it.factor_ *= factor;
        return it;
    }

    friend indexed_tensor<Derived,T> operator*(const T factor, const indexed_tensor<Derived,T>& other)
    {
        return other*factor;
    }

    indexed_tensor<Derived,T>& operator*=(const T factor)
    {
        tensor_.scale(factor, idx_);
        return *this;
    }

    indexed_tensor<Derived,T>& operator=(const T val)
    {
        Derived tensor(tensor_, val);
        *this = tensor[""];
        return *this;
    }

    indexed_tensor<Derived,T>& operator+=(const T val)
    {
        Derived tensor(tensor_, val);
        *this += tensor[""];
        return *this;
    }

    indexed_tensor<Derived,T>& operator-=(const T val)
    {
        Derived tensor(tensor_, val);
        *this -= tensor[""];
        return *this;
    }
};

template <class Derived1, class Derived2, class T>
//typename std::enable_if<std::is_same<const Derived1, const Derived2>::value,IndexedTensorMult<Derived1,T> >::type
indexed_tensor_mult<Derived1,T>
operator*(const indexable_tensor_base<Derived1,T>& t1, const indexed_tensor<Derived2,T>& t2)
{
    return indexed_tensor_mult<Derived1,T>(t1[t1.implicit()], t2);
}

template <class Derived1, class Derived2, class T>
//typename std::enable_if<std::is_same<const Derived1, const Derived2>::value,IndexedTensorMult<Derived1,T> >::type
indexed_tensor_mult<Derived1,T>
operator*(const ScaledTensor<Derived1,T>& t1, const indexed_tensor<Derived2,T>& t2)
{
    Derived1& A = t1.tensor_.get_derived();

    return indexed_tensor_mult<Derived1,T>(A[A.implicit()]*t1.factor_, t2);
}

template <typename Derived, typename T>
struct indexed_tensor_mult
{
private:
    const indexed_tensor_mult& operator=(const indexed_tensor_mult<Derived, T>& other);

public:
    indexed_tensor<const Derived,T> A_;
    indexed_tensor<const Derived,T> B_;
    T factor_;

    template <class Derived1, class Derived2>
    indexed_tensor_mult(const indexed_tensor<Derived1,T>& A, const indexed_tensor<Derived2,T>& B)
        : A_(A), B_(B), factor_(A.factor_*B.factor_) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    indexed_tensor_mult<Derived,T> operator-() const
    {
        indexed_tensor_mult<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    indexed_tensor_mult<Derived,T> operator*(const T factor) const
    {
        indexed_tensor_mult<Derived,T> ret(*this);
        ret.factor_ *= factor;
        return ret;
    }

    indexed_tensor_mult<Derived,T> operator/(const T factor) const
    {
        indexed_tensor_mult<Derived,T> ret(*this);
        ret.factor_ /= factor;
        return ret;
    }

    friend indexed_tensor_mult<Derived,T> operator*(const T factor, const indexed_tensor_mult<Derived,T>& other)
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
template <class Derived, typename T>
T scalar(const tensor::indexed_tensor_mult<Derived,T>& itm)
{
    return itm.factor_*itm.B_.tensor_.dot(itm.A_.tensor_, itm.A_.idx_,
                                                          itm.B_.idx_);
}

} // namespace ambit

#endif
