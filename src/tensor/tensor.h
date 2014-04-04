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

template <class Derived, typename T> struct tensor;
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
    using ambit::tensor::tensor< Derived, T >::get_derived; \
    using ambit::tensor::tensor< Derived, T >::operator=; \
    using ambit::tensor::tensor< Derived,T >::operator+=; \
    using ambit::tensor::tensor< Derived,T >::operator-=; \
    using ambit::tensor::tensor< Derived,T >::operator*=; \
    using ambit::tensor::tensor< Derived,T >::operator/=; \
    using ambit::tensor::tensor< Derived,T >::operator*; \
    using ambit::tensor::tensor< Derived,T >::operator/; \
    Derived & operator=(const Derived & other) \
    { \
        sum((T)1, false, other, (T)0); \
        return *this; \
    } \
    private:

template<class Derived, typename T>
struct tensor
{
    typedef T dtype;
    std::string name;

    tensor(const std::string& name) : name(name) {}
    virtual ~tensor() {}

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
        mult((T)1, get_derived(), other, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(cvDerived& other)
    {
        div((T)1, get_derived(), other, (T)0);
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
        mult(other.factor_, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(const ScaledTensor<cvDerived,T>& other)
    {
        div((T)1/other.factor_, get_derived(), other.tensor_, (T)0);
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
        div(other.factor_, get_derived(), other.tensor_, (T)0);
        return get_derived();
    }

    template <typename cvDerived>
    Derived& operator/=(const InvertedTensor<cvDerived,T>& other)
    {
        mult((T)1/other.factor_, get_derived(), other.tensor_, (T)0);
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
        tensor_.sum((T)1, other, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator+=(const cvDerived& other)
    {
        tensor_.sum((T)1, other, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator-=(const cvDerived& other)
    {
        tensor_.sum((T)(-1), other, factor_);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator*=(const cvDerived& other)
    {
        tensor_.mult(factor_, tensor_, other, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const cvDerived& other)
    {
        tensor_.div(factor_, tensor_, other, (T)0);
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
        tensor_.mult(factor_*other.factor_, tensor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const ScaledTensor<cvDerived,T>& other)
    {
        tensor_.div(factor_/other.factor_, tensor_, other.tensor_, (T)0);
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
        tensor_.div(factor_*other.factor_, tensor_, other.tensor_, (T)0);
        return *this;
    }

    template <typename cvDerived>
    ScaledTensor<Derived,T>& operator/=(const InvertedTensor<cvDerived,T>& other)
    {
        tensor_.mult(factor_/other.factor_, tensor_, other.tensor_, (T)0);
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
class IndexNotFoundError : public TensorError
{
public:
    virtual const char* what() const throw() { return "index not found."; }
};
class IndexAlreadyExistsError : public TensorError
{
public:
    virtual const char *what() const throw() { return "index already exists in global set."; }
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

/**
 * Main interface definitions
 *
 * In all cases, a repeated index label, either within the same tensor or in more than one tensor requires that the edge length be the same
 * in all labeled indices. With the exception of the resym operation and operations supporting it, the symmetry relation must also stay the same.
 * However, even in this case the symmetry relation may not change from AS or SH to SY or vice versa. 0-dimension tensors (scalars) are
 * allowed in all functions. In this case, the arguments len_*, sym_*, and idx_* may be NULL and will not be referenced. The data array for a scalar
 * must NOT be NULL, and should be of size 1. The special case of beta == +/-0.0 will overwrite special floating point values such as NaN and Inf
 * in the output.
 */

/**
 * Binary operations: C = alpha*A*B + beta*C
 */
typedef int (*tensor_func_binary)(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                                                      const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B,
                                  const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* sym_C, const int* idx_C);

/**
 * Multiply two tensors together and sum onto a third
 *
 * This form generalizes contraction and weighting with the unary operations trace, transpose, resym, diagonal, and replicate. Note that
 * the binary contraction operation is similar in form to the unary trace operation, while the binary weighting operation is similar in form to the
 * unary diagonal operation. Any combination of these operations may be performed. Even in the case that only a subset of the elements of C are written
 * to by the multiplication, all elements of C are first scaled by beta. Replication is performed in-place.
 */
int tensor_mult_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                                     const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B,
                 const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* sym_C, const int* idx_C);

/**
 * Contract two tensors into a third
 *
 * The general form for a contraction is ab...ef... * ef...cd... -> ab...cd... where the indices ef... will be summed over.
 * Indices may be transposed in any tensor, but the symmetry relations must not be changed, with the exception that ab... may or may not
 * be (anti)symmetrized with cd.... Any index group may be empty (in the case that ef... is empty, this reduces to an outer product).
 */
int tensor_contract_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                                         const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B,
                     const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* sym_C, const int* idx_C);

/**
 * Weight a tensor by a second and sum onto a third
 *
 * The general form for a weighting is ab...ef... * ef...cd... -> ab...cd...ef... with no indices being summed over.
 * Indices may be transposed in any tensor, but the symmetry relations must not be changed, with the addition that ab... must
 * be (anti)symmetrized with cd.... as ab... and cd... are with ef.... Any index group may be empty
 * (in the case that ef... is empty, this reduces to an outer product).
 */
int tensor_weight_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                                       const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B,
                   const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* sym_C, const int* idx_C);

/**
 * Sum the outer product of two tensors onto a third
 *
 * The general form for an outer product is ab... * cd... -> ab...cd... with no indices being summed over.
 * Indices may be transposed in any tensor, but the symmetry relations must not be changed, with the exception that ab... may or may not
 * be (anti)symmetrized with cd....
 */
int tensor_outer_prod_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                                           const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B,
                       const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* sym_C, const int* idx_C);

typedef int (*tensor_func_binary_dense)(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                                            const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B,
                                        const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* idx_C);

int tensor_mult_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                           const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B,
                       const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* idx_C);

int tensor_contract_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                               const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B,
                           const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* idx_C);

int tensor_weight_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                             const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B,
                         const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* idx_C);

int tensor_outer_prod_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                                 const double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B,
                             const double beta,        double* C, const int ndim_C, const int* len_C, const int* ldc, const int* idx_C);

/**
 * Unary operations: B = alpha*A + beta*B
 */
typedef int (*tensor_func_unary)(const double alpha, const double* A, const int ndim_A, const int* lda, const int* len_A, const int* sym_A, const int* idx_A,
                                 const double beta,        double* B, const int ndim_B, const int* ldb, const int* len_B, const int* sym_B, const int* idx_B);

/**
 * sum a tensor (presumably operated on in one or more ways) onto a second
 *
 * This form generalizes all of the unary operations trace, transpose, resym, diagonal, and replicate, which may be performed
 * in any combination. Even in the case that only a subset of the elements of B are written to by the operation, all elements
 * of B are first scaled by beta. Replication is performed in-place.
 */
int tensor_sum_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

/**
 * Transpose a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> P(ab...) where P is some permutation. Transposition may change
 * the order in which the elements of the tensor are physically stored, with the exception that (anti)symmetric
 * groups of indices must remain together. The symmetry relations of the indices must not be changed. A transposition among
 * symmetric indices produces no effect, while a transposition among antisymmetric indices may induce a change in sign,
 * but no physical reordering of elements.
 */
int tensor_transpose_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                      const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

/**
 * Change the symmetry relations of a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> ab... with a different set of symmetry relations in A and B (this means
 * also that A and B will not be the same size in general). Indices may either be unpacked or (anti)symmetrized by this operation, with
 * both possibilities allowed concurrently on disjoint sets of indices. A set of (anti)symmetric indices may be partially unpacked, leaving
 * two or more indices in packed storage, while one or more nonsymmetric indices may be (anti)symmetrized onto an existing packed set of
 * indices. Even in the case that only a subset of the elements of B are written to by the operation, all elements of B are first scaled
 * by beta. Transposition of the indices is allowed. Each (partial) unpacking includes a permutation factor of n'!/n! where n and n' are
 * number of (anti)symmetric indices before and after the unpacking.
 */
int tensor_resym_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                  const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

/**
 * Sum over (semi)diagonal elements of a tensor and sum onto a second
 *
 * The general form for a trace operation is ab...k*l*... -> ab... where k* denotes the index k appearing one or more times, etc. and where
 * the indices kl... will be summed (traced) over. Indices may be transposed (except for (anti)symmetric groups), and multiple appearances
 * of the traced indices kl... need not appear together. Either set of indices may be empty, with the special case that when no indices
 * are traced over, the result is the same as transpose. A trace over two or more antisymmetric indices produces no effect, except that B
 * is still scaled by beta.
 */
int tensor_trace_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                  const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

/**
 * Replicate a tensor and sum onto a second
 *
 * The general form for a replication operation is ab... -> ab...c*d*... where c* denotes the index c appearing one or more times.
 * Any indices may be transposed, with the exception that (anti)symmetric groups of indices must stay together. Replication is
 * performed in-place (meaning that no additional scratch space is required). Even in the case that only a subset of the elements of
 * B are written two, all elements of B are first scaled by beta.
 */
int tensor_replicate_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                      const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

/**
 * Extract the (semi)diagonal of a tensor and sum onto a second
 *
 * The general form for a diagonal extraction is ab...k*l*... -> ab...k*l*... where k* denotes the index k appearing one or more times
 * (which need not be the same number of times in A and B). Extracting the diagonal of a group of antisymmetric indices produces no effect,
 * except that B is still scaled by beta. Indices may be transposed, except (anti)symmetric groups of indices must stay together.
 */
int tensor_diagonal_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A,
                     const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* sym_B, const int* idx_B);

typedef int (*tensor_func_unary_dense)(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                                       const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

int tensor_sum_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                      const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

int tensor_transpose_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                            const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

int tensor_trace_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                        const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

int tensor_replicate_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                            const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

int tensor_diagonal_dense_(const double alpha, const double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A,
                           const double beta,        double* B, const int ndim_B, const int* len_B, const int* ldb, const int* idx_B);

/*
 * Helper function definitions
 */

int tensor_scale_(const double alpha, double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A, const int* idx_A);

int tensor_scale_dense_(const double alpha, double* A, const int ndim_A, const int* len_A, const int* lda, const int* idx_A);

int tensor_slice_dense(const double*  A, const int  ndim_A, const int* len_A, const int* lda,
                             double** B,       int* ndim_B,       int* len_B,       int* ldb,
                       const int* start, const int* len);

/**
 * Print a tensor in the form:
 *
 * i_0;k i_1;k ... i_ndim;k A_k
 *
 * where i_j;k are the index string for the kth element of A (A_k) in packed order, for k = 0...size-1.
 */
int tensor_print(const double* A, const int ndim_A, const int* len_A, const int* lda, const int* sym_A);

/**
 * Print a tensor in the form:
 *
 * i_0;k i_1;k ... i_ndim;k A_k
 *
 * where i_j;k are the index string for the kth element of A (A_k) in order, for k = 0...size-1.
 */
int tensor_print_dense(const double* A, const int ndim_A, const int* len_A, const int* lda);

/**
 * Calculate the number of non-zero (and hence stored) elements in the given tensor.
 */
int64_t tensor_size(const int ndim, const int* len, const int* ld, const int* sym);

/**
 * Calculate the number of elements in the given tensor, assuming a dense (rectangular) layout.
 */
int64_t tensor_size_dense(const int ndim, const int* len, const int* ld);

/**
 * Combine each group of (anti)symmetric indices in the given tensor into a single index of the same total length.
 */
int tensor_densify(int* ndim, int* len, const int* sym);

/**
 * Take a packed (anti)symmetric tensor and expand it to a dense (rectangular) layout, with all redundant and
 * zero elements written explicitly. The result is scaled by a factor prod_k 1/[(l_k)^(n_k)] for k=0...ngroup, where
 * l_k is the length of the indices in the kth (anti)symmetric group of indices and n_k is the number of such indices.
 * This factor produces a normalized result when combined with tensor_symmetrize or tensor_pack.
 */
int tensor_unpack(const double* A, double* B, const int ndim_A, const int* len_A, const int* sym_A);

/**
 * Perform the reverse of tensor_unpack, referencing only non-redundant, non-zero elements of A (although which set of
 * redundant elements will be accessed is unspecified). The result is scaled by a factor prod_k (l_k)^(n_k) for
 *  k=0...ngroup, where l_k is the length of the indices in the kth (anti)symmetric group of indices and n_k is the
 *  number of such indices. The behavior is the same as tensor_symmetrize for the same input, except that fewer elements
 *  must be accessed.
 */
int tensor_pack(const double* A, double* B, const int ndim_A, const int* len_A, const int* sym_A);

/**
 * Perform an explicit (anti)symmetrization of the given dense tensor.
 */
int tensor_symmetrize(const double* A, double* B, const int ndim_A, const int* len_A, const int* sym_A);

} // namespace tensor

template <class Derived, typename T>
T scalar(const tensor::TensorMult<Derived, T>& tm)
{
    return tm.factor_*tm.B_.tensor_.dot(tm.A_.tensor_);
}

} // namespace ambit

#endif
