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

#if !defined(AMBIT_LIB_TENSOR_TENSOR)
#define AMBIT_LIB_TENSOR_TENSOR

#include <stdexcept>
#include <string>
#include <complex>

/*
 * Symmetry types
 */
enum kTensorSymmetryTypes {
    kTensorSymmetryNonSymmetric = 0,
    kTensorSymmetrySymmetric = 1,
    kTensorSymmetryAntiSymmetric = 2,
    kTensorSymmetrySymmetricHollow = 3
};

/*
 * Shorthand notation for symmetry types
 */
#ifndef NS
#define NS  kTensorSymmetryNonSymmetric
#define SY  kTensorSymmetrySymmetric
#define AS  kTensorSymmetryAntiSymmetric
#define SH  kTensorSymmetrySymmetricHollow
#endif

/*
 * Return codes
 */
enum kTensorReturnCodes {
    kTensorReturnCodeSuccess = 0,
    kTensorReturnCodeOutOfBounds = -1,
    kTensorReturnCodeLengthMismatch = -2,
    kTensorReturnCodeIndexMismatch = -3,
    kTensorReturnCodeInvalidNDim = -4,
    kTensorReturnCodeInvalidLength = -5,
    kTensorReturnCodeInvalidLD = -6,
    kTensorReturnCodeLDTooSmall = -7,
    kTensorReturnCodeSymmetryMismatch = -8,
    kTensorReturnCodeInvalidSymmetry = -9,
    kTensorReturnCodeInvalidStart = -10
};

namespace ambit {

template <typename T>
struct real_type
{
    typedef T type;
};

template <typename T>
struct real_type<std::complex<T> >
{
    typedef T type;
};

namespace tensor {

template <class Derived, typename T> struct Tensor;
template <class Derived, typename T> struct ScaledTensor;
template <class Derived, typename T> struct InvertedTensor;
template <class Derived, typename T> struct TensorMult;
template <class Derived, typename T> struct TensorDiv;

class TensorError;
class OutOfBoundsError;
class LengthMismatchError;
class IndexMismatchError;
class InvalidNdimError;
class InvalidLengthError;
class InvalidLdError;
class LdTooSmallError;
class SymmetryMismatchError;
class InvalidSymmetryError;
class InvalidStartError;

#define INSTANTIATE_SPECIALIZATIONS(name) \
template class name<double>;

#define INSTANTIATE_SPECIALIZATIONS_2(name,extra1) \
template class name<double,extra1>;

#define INSTANTIATE_SPECIALIZATIONS_3(name,extra1,extra2) \
template class name<double,extra1,extra2>;

#define INHERIT_FROM_TENSOR(Derived,T) \
    public: \
    using ambit::tensor::Tensor< Derived, T >::get_derived; \
    using ambit::tensor::Tensor< Derived, T >::operator=; \
    using ambit::tensor::Tensor< Derived,T >::operator+=; \
    using ambit::tensor::Tensor< Derived,T >::operator-=; \
    using ambit::tensor::Tensor< Derived,T >::operator*=; \
    using ambit::tensor::Tensor< Derived,T >::operator/=; \
    using ambit::tensor::Tensor< Derived,T >::operator*; \
    using ambit::tensor::Tensor< Derived,T >::operator/; \
    Derived & operator=(const Derived & other) \
    { \
        sum((T)1, false, other, (T)0); \
        return *this; \
    } \
    private:

template<class Derived, typename T>
struct Tensor
{
    typedef T dtype;
    std::string name;

    Tensor() {}
    Tensor(const std::string& name) : name(name) {}
    virtual ~Tensor() {}

    const std::string& get_name() const { return name; }

    Derived& get_derived() { return static_cast<Derived&>(*this); }
    const Derived& get_derived() const { return static_cast<const Derived&>(*this); }

    /**********************************************************************
     *
     * Operators with scalars
     *
     *********************************************************************/
    Derived& operator=(const T val)
    {
        sum(val, (T)0);
        return get_derived();
    }

    Derived& operator+=(const T val)
    {
        sum(val, (T)1);
        return get_derived();
    }

    Derived& operator*=(const T val)
    {
        mul(val);
        return get_derived();
    }

    Derived& operator/=(const T val)
    {
        mult(1.0/val);
        return get_derived();
    }

    /**********************************************************************
     *
     * Binary operations (multiplication and division)
     *
     *********************************************************************/
    template<typename cvDerived>
    Derived& operator=(const TensorMult<cvDerived,T>& other)
    {
        mult(other.factor_, other.A_.tensor_,
                            other.B_.tensor_, (T)0);
        return get_derived();
    }

    template<typename cvDerived>
    Derived& operator+=(const TensorMult<cvDerived,T>& other)
    {
        mult(other.factor_, other.A_.tensor_,
                            other.B_.tensor_, (T)1);
        return get_derived();
    }

    template<typename cvDerived>
    Derived& operator-=(const TensorMult<cvDerived,T>& other)
    {
        mult(-other.factor_, other.A_.tensor_,
                             other.B_.tensor_, (T)1);
        return get_derived();
    }

    template<typename cvDerived>
    Derived& operator=(const TensorDiv<cvDerived,T>& other)
    {
        div(other.factor_, other.A_.tensor_,
                           other.B_.tensor_, (T)0);
        return get_derived();
    }

    template<typename cvDerived>
    Derived& operator+=(const TensorDiv<cvDerived,T>& other)
    {
        div(other.factor_, other.A_.tensor_,
                           other.B_.tensor_, (T)1);
        return get_derived();
    }

    template<typename cvDerived>
    Derived& operator-=(const TensorDiv<cvDerived,T>& other)
    {
        div(-other.factor_, other.A_.tensor_,
                            other.B_.tensor_, (T)1);
        return get_derived();
    }

    /**********************************************************************
     *
     * Unary operations (assignment, summation, multiplication, and division)
     *
     *********************************************************************/
    Derived& operator=(const Derived& other)
    {
        sum((T)1, false, other, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator=(cvDerived& other)
    {
        sum((T)1, false, other, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator+=(cvDerived& other)
    {
        sum((T)1, false, other, (T)1);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator-=(cvDerived& other)
    {
        sum((T)(-1), false, other, (T)1);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator*=(cvDerived& other)
    {
        mult((T)1, false, get_derived(), false, other, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(cvDerived& other)
    {
        div((T)1, false, get_derived(), false, other, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator=(const ScaledTensor<cvDerived,T>& other)
    {
        sum(other.factor_, other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator+=(const ScaledTensor<cvDerived,T>& other)
    {
        sum(other.factor_, other.tensor_, (T)1);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator-=(const ScaledTensor<cvDerived,T>& other)
    {
        sum(-other.factor_, other.tensor_, (T)1);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator*=(const ScaledTensor<cvDerived,T>& other)
    {
        mult(other.factor_, false, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(const ScaledTensor<cvDerived,T>& other)
    {
        div((T)1/other.factor_, false, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator=(const InvertedTensor<cvDerived,T>& other)
    {
        invert(other.factor_, other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator+=(const InvertedTensor<cvDerived,T>& other)
    {
        invert(other.factor_, other.tensor_, (T)1);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator-=(const InvertedTensor<cvDerived,T>& other)
    {
        invert(-other.factor_, other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator*=(const InvertedTensor<cvDerived,T>& other)
    {
        div(other.factor_, false, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(const InvertedTensor<cvDerived,T>& other)
    {
        mult((T)1/other.factor_, false, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    /**********************************************************************
     *
     * Intermediate operations
     *
     *********************************************************************/
    friend ScaledTensor<Derived,T> operator*(const T factor, Derived& other)
    {
        return ScaledTensor<Derived,T>(other.get_derived(), factor);
    }

    friend ScaledTensor<const Derived,T> operator*(const T factor, const Derived& other)
    {
        return ScaledTensor<const Derived,T>(other.get_derived(), factor);
    }

    ScaledTensor<Derived,T> operator*(const T factor)
    {
        return ScaledTensor<Derived,T>(get_derived(), factor);
    }

    ScaledTensor<const Derived,T> operator*(const T factor) const
    {
        return ScaledTensor<const Derived,T>(get_derived(), factor);
    }

    friend InvertedTensor<const Derived,T> operator/(const T factor, const Derived& other)
    {
        return InvertedTensor<const Derived,T>(other.get_derived(), factor);
    }

    ScaledTensor<Derived,T> operator/(const T factor)
    {
        return ScaledTensor<Derived,T>(get_derived(), (T)1/factor);
    }

    ScaledTensor<const Derived,T> operator/(const T factor) const
    {
        return ScaledTensor<const Derived,T>(get_derived(), (T)1/factor);
    }

    ScaledTensor<Derived,T> operator-()
    {
        return ScaledTensor<Derived,T>(get_derived(), (T)(-1));
    }

    ScaledTensor<const Derived,T> operator-() const
    {
        return ScaledTensor<const Derived,T>(get_derived(), (T)(-1));
    }

    template <typename cvDerived>
    TensorMult<Derived,T> operator*(const cvDerived& other) const
    {
        return TensorMult<Derived,T>(ScaledTensor<const Derived,T>(get_derived(), (T)1),
                                      ScaledTensor<const Derived,T>(other.get_derived(), (T)1));
    }

    template <typename cvDerived>
    TensorDiv<Derived,T> operator/(const cvDerived& other) const
    {
        return TensorDiv<Derived,T>(ScaledTensor<const Derived,T>(get_derived(), (T)1),
                                     ScaledTensor<const Derived,T>(other.get_derived(), (T)1));
    }

    /**********************************************************************
     *
     * Stubs
     *
     *********************************************************************/

    /*
     * this = alpha*this + beta*A*B
     */
    virtual void mult(const T alpha, const Derived& A,
                                     const Derived& B, const T beta) = 0;

    /*
     * this = alpha*this
     */
    virtual void mult(const T alpha) = 0;

    /*
     * this = alpha*this + beta*A/B
     */
    virtual void div(const T alpha, const Derived& A,
                                    const Derived& B, const T beta) = 0;

    /*
     * this = alpha*this + beta*A
     */
    virtual void sum(const T alpha, const Derived& A, const T beta) = 0;

    /*
     * this = alpha*this + beta
     */
    virtual void sum(const T alpha, const T beta) = 0;

    /*
     * this = alpha*this + beta/A
     */
    virtual void invert(const T alpha, const Derived& A, const T beta) = 0;

    /*
     * scalar = A*this
     */
    virtual T dot(const Derived& A) const = 0;
};

template <class Derived, typename T>
struct ScaledTensor
{
    Derived& tensor_;
    T factor_;

    template <typename cvDerived>
    ScaledTensor(const ScaledTensor<cvDerived,T>& other)
        : tensor_(other.tensor_), factor_(other.factor_) {}

    ScaledTensor(Derived& tensor, const T factor)
        : tensor_(tensor), factor_(factor) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    ScaledTensor<Derived,T> operator-() const
    {
        ScaledTensor<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Unary tensor operations
     *
     *********************************************************************/
    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator=(const cvDerived& other)
    {
        tensor_.sum((T)1, false, other, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const cvDerived& other)
    {
        tensor_.sum((T)1, false, other, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const cvDerived& other)
    {
        tensor_.sum((T)(-1), false, other, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator*=(const cvDerived& other)
    {
        tensor_.mult(factor_, false, tensor_, false, other, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const cvDerived& other)
    {
        tensor_.div(factor_, false, tensor_, false, other, (T)0);
        return *this;
    }

    ScaledTensor<Derived,T>& operator=(const ScaledTensor<Derived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.sum(other.factor_, other.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.sum(-other.factor_, other.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator*=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.mult(factor_*other.factor_, false, tensor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.div(factor_/other.factor_, false, tensor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.invert(other.factor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.invert(other.factor_, other.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.invert(-other.factor_, other.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator*=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.div(factor_*other.factor_, false, tensor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.mult(factor_/other.factor_, false, tensor_, other.tensor_, (T)0);
        return *this;
    }

    /**********************************************************************
     *
     * Binary tensor operations
     *
     *********************************************************************/
    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator=(const TensorMult<cvDerived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.B_.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const TensorMult<cvDerived,T>& other)
    {
        tensor_.mult(other.factor_, other.A_.tensor_, other.B_.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const TensorMult<cvDerived,T>& other)
    {
        tensor_.mult(-other.factor_, other.A_.tensor_, other.B_.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator=(const TensorDiv<cvDerived,T>& other)
    {
        tensor_.div(other.factor_, other.A_.tensor_, other.B_.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const TensorDiv<cvDerived,T>& other)
    {
        tensor_.div(other.factor_, other.A_.tensor_, other.B_.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const TensorDiv<cvDerived,T>& other)
    {
        tensor_.div(-other.factor_, other.A_.tensor_, other.B_.tensor_, factor_);
        return *this;
    }

    template <typename cvDerived>
    TensorMult<Derived,T> operator*(const ScaledTensor<cvDerived,T>& other) const
    {
        return TensorMult<Derived,T>(*this, other);
    }

    template <typename cvDerived>
    TensorMult<Derived,T> operator*(const cvDerived& other) const
    {
        return TensorMult<Derived,T>(*this, ScaledTensor<const Derived,T>(other.get_derived(), (T)1));
    }

    template <typename cvDerived>
    TensorDiv<Derived,T> operator/(const ScaledTensor<cvDerived,T>& other) const
    {
        return TensorDiv<Derived,T>(*this, other);
    }

    template <typename cvDerived>
    TensorDiv<Derived,T> operator/(const cvDerived& other) const
    {
        return TensorDiv<Derived,T>(*this, ScaledTensor<const Derived,T>(other.get_derived(), (T)1));
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    ScaledTensor<Derived,T> operator*(const T factor) const
    {
        ScaledTensor<Derived,T> it(*this);
        it.factor_ *= factor;
        return it;
    }

    friend ScaledTensor<Derived,T> operator*(const T factor, const ScaledTensor<Derived,T>& other)
    {
        return other*factor;
    }

    ScaledTensor<Derived,T> operator/(const T factor) const
    {
        ScaledTensor<Derived,T> it(*this);
        it.factor_ /= factor;
        return it;
    }

    friend InvertedTensor<Derived,T> operator/(const T factor, const ScaledTensor<Derived,T>& other)
    {
        return InvertedTensor<Derived,T>(other.tensor_, factor/other.factor_);
    }

    ScaledTensor<Derived,T>& operator=(const T val)
    {
        tensor_.sum(val, (T)0);
        return *this;
    }

    ScaledTensor<Derived,T>& operator+=(const T val)
    {
        tensor_.sum(val, factor_);
        return *this;
    }

    ScaledTensor<Derived,T>& operator-=(const T val)
    {
        tensor_.sum(-val, factor_);
        return *this;
    }

    ScaledTensor<Derived,T>& operator*=(const T val)
    {
        tensor_.mult(val);
        return *this;
    }

    ScaledTensor<Derived,T>& operator/=(const T val)
    {
        tensor_.mult((T)1/val);
        return *this;
    }
};

template <class Derived1, class Derived2, class T>
TensorMult<Derived1,T> operator*(const Derived1& t1, const ScaledTensor<Derived2,T>& t2)
{
    return TensorMult<Derived1,T>(ScaledTensor<const Derived1,T>(t1.get_derived(), (T)1), t2);
}

template <class Derived1, class Derived2, class T>
TensorDiv<Derived1,T> operator/(const Derived1& t1, const ScaledTensor<Derived2,T>& t2)
{
    return TensorDiv<Derived1,T>(ScaledTensor<const Derived1,T>(t1.get_derived(), (T)1), t2);
}

template <class Derived, typename T>
class InvertedTensor
{
private:
    const InvertedTensor& operator=(const InvertedTensor<Derived,T>& other);

public:
    Derived& tensor_;
    T factor_;

    InvertedTensor(Derived& tensor, const T factor)
        : tensor_(tensor), factor_(factor) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    InvertedTensor<Derived,T> operator-() const
    {
        InvertedTensor<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return *this;
    }

    /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/
    InvertedTensor<Derived,T> operator*(const T factor) const
    {
        InvertedTensor<Derived,T> ret(*this);
        ret.factor_ *= factor;
        return ret;
    }

    InvertedTensor<Derived,T> operator/(const T factor) const
    {
        InvertedTensor<Derived,T> ret(*this);
        ret.factor_ /= factor;
        return ret;
    }

    friend InvertedTensor<Derived,T> operator*(const T factor, const InvertedTensor<Derived,T>& other)
    {
        return other*factor;
    }
};

template <class Derived, typename T>
class TensorMult
{
private:
    const TensorMult& operator=(const TensorMult<Derived,T>& other);

public:
    ScaledTensor<const Derived,T> A_;
    ScaledTensor<const Derived,T> B_;
    T factor_;

    template <class Derived1, class Derived2>
    TensorMult(const ScaledTensor<Derived1,T>& A, const ScaledTensor<Derived2,T>& B)
        : A_(A), B_(B), factor_(A.factor_*B.factor_) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    TensorMult<Derived,T> operator-() const
    {
        TensorMult<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    TensorMult<Derived,T> operator*(const T factor) const
    {
        TensorMult<Derived,T> ret(*this);
        ret.factor_ *= factor;
        return ret;
    }

    TensorMult<Derived,T> operator/(const T factor) const
    {
        TensorMult<Derived,T> ret(*this);
        ret.factor_ /= factor;
        return ret;
    }

    friend TensorMult<Derived,T> operator*(const T factor, const TensorMult<Derived,T>& other)
    {
        return other*factor;
    }
};

template <class Derived, typename T>
class TensorDiv
{
private:
    const TensorDiv& operator=(const TensorDiv<Derived,T>& other);

public:
    ScaledTensor<const Derived,T> A_;
    ScaledTensor<const Derived,T> B_;
    T factor_;

    template <class Derived1, class Derived2>
    TensorDiv(const ScaledTensor<Derived1,T>& A, const ScaledTensor<Derived2,T>& B)
        : A_(A), B_(B), factor_(A.factor_/B.factor_) {}

    /**********************************************************************
     *
     * Unary negation
     *
     *********************************************************************/
    TensorDiv<Derived,T> operator-() const
    {
        TensorDiv<Derived,T> ret(*this);
        ret.factor_ = -ret.factor_;
        return ret;
    }

    /**********************************************************************
     *
     * Operations with scalars
     *
     *********************************************************************/
    TensorDiv<Derived,T> operator*(const T factor) const
    {
        TensorDiv<Derived,T> ret(*this);
        ret.factor_ *= factor;
        return ret;
    }

    TensorDiv<Derived,T> operator/(const T factor) const
    {
        TensorDiv<Derived,T> ret(*this);
        ret.factor_ /= factor;
        return ret;
    }

    friend TensorDiv<Derived,T> operator*(const T factor, const TensorDiv<Derived,T>& other)
    {
        return other*factor;
    }
};

class TensorError : public std::exception
{
    public:
        virtual const char* what() const throw() = 0;
};
class OutOfBoundsError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "out-of-bounds read or write"; }
};
class LengthMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "length mismatch error"; }
};
class IndexMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "index mismatch error"; }
};
class InvalidNdimError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid number of dimensions"; }
};
class InvalidLengthError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid length"; }
};
class InvalidLdError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid leading dimension"; }
};
class LdTooSmallError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "leading dimension is too small"; }
};
class SymmetryMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "symmetry mismatch error"; }
};
class InvalidSymmetryError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid symmetry value"; }
};
class InvalidStartError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid start value"; }
};

} // namespace tensor

template <class Derived, typename T>
T scalar(const tensor::TensorMult<Derived, T>& tm)
{
    return tm.factor_*tm.B_.tensor_.dot(tm.A_.tensor_);
}

} // namespace ambit

#endif
