/*
 * Copyright (C) 2013 Devin Matthews
 *
 * This is a slimmed down version of the tensor framework developed by
 * Devin Matthews. The version by Devin was tied to ambit. This
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

#if !defined(AMBIT_TENSOR_COMPOSITE_TENSOR_H)
#define AMBIT_TENSOR_COMPOSITE_TENSOR_H

#include <vector>
#include <string>
#include <algorithm>

#include "util.h"
#include "indexable_tensor.h"

namespace ambit
{
namespace tensor
{

#define INHERIT_FROM_COMPOSITE_TENSOR(Derived,Base,T) \
    protected: \
    using ambit::tensor::composite_tensor< Derived, Base, T >::tensors; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::add_tensor; \
    public: \
    using ambit::tensor::composite_tensor< Derived, Base, T >::mult; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::div; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::sum; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::invert; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::dot; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::exists; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::operator(); \
    using ambit::tensor::tensor< Derived,T >::get_derived; \
    using ambit::tensor::tensor< Derived,T >::operator=; \
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

#define INHERIT_FROM_INDEXABLE_COMPOSITE_TENSOR(Derived,Base,T) \
    protected: \
    using ambit::tensor::composite_tensor< Derived, Base, T >::tensors; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::add_tensor; \
    using ambit::tensor::indexable_tensor_base< Derived, T >::ndim; \
    public: \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::mult; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::sum; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::div; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::invert; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::scale; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::dot; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::operator=; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::operator+=; \
    using ambit::tensor::indexable_composite_tensor< Derived, Base, T >::operator-=; \
    using ambit::tensor::indexable_tensor_base< Derived, T >::operator[]; \
    using ambit::tensor::composite_tensor< Derived, Base, T >::operator(); \
    using ambit::tensor::composite_tensor< Derived, Base, T >::exists; \
    using ambit::tensor::tensor< Derived,T >::get_derived; \
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

template <class Derived, class Base, class T>
class composite_tensor : public tensor<Derived,T>
{
protected:
    struct TensorRef
    {
        Base* tensor;
        bool isAlloced;
        int ref;
        TensorRef(Base* tensor_=NULL, bool isAlloced=false, int ref=-1)
            : tensor(tensor_), isAlloced(isAlloced), ref(ref) {}
        bool operator==(const Base* other) const { return tensor == other; }
        bool operator!=(const Base* other) const { return tensor != other; }
    };

    std::vector<TensorRef> tensors;

    Base& add_tensor(Base* new_tensor, bool isAlloced=true)
    {
        tensors.push_back(TensorRef(new_tensor, isAlloced));
        return *new_tensor;
    }

    Base& add_tensor(Base& new_tensor, bool isAlloced=false)
    {
        tensors.push_back(TensorRef(&new_tensor, isAlloced));
        return new_tensor;
    }

    Base& add_tensor(int ref)
    {
        assert(ref >= -1 && ref < tensors.size());
        tensors.push_back(TensorRef(tensors[ref].tensor, false, ref));
        return *tensors[ref].tensor;
    }

public:
    composite_tensor(const composite_tensor<Derived,Base,T>& other)
        : tensor<Derived,T>(other.name), tensors(other.tensors)
    {
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1)
            {
                tensors[i].tensor = new Base(*tensors[i].tensor);
            }
        }
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i].ref != -1)
            {
                tensors[i].tensor = tensors[tensors[i].ref].tensor;
            }
        }
    }

    composite_tensor(const std::string& name, const composite_tensor<Derived,Base,T>& other)
        : tensor<Derived,T>(name), tensors(other.tensors)
    {
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1)
            {
                tensors[i].tensor = new Base(name, *tensors[i].tensor);
            }
        }
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i].ref != -1)
            {
                tensors[i].tensor = tensors[tensors[i].ref].tensor;
            }
        }
    }

    composite_tensor(const std::string& name, int ntensors = 0)
        : tensor<Derived,T>(name), tensors(ntensors) {}

    virtual ~composite_tensor()
    {
        for (int i = tensors.size()-1;i >= 0;i--)
        {
            if (tensors[i].isAlloced)
            {
                delete tensors[i].tensor;
            }
        }
    }

    int getNumTensors() const { return tensors.size(); }

    bool exists(int idx) const
    {
        return tensors[idx] != NULL;
    }

    /**********************************************************************
         *
         * Subtensor indexing
         *
         *********************************************************************/
    Base& operator()(int idx)
    {
        if (tensors[idx] == NULL)
            throw std::logic_error("tensor component does not exist");
        return *tensors[idx].tensor;
    }

    const Base& operator()(int idx) const
    {
        if (tensors[idx] == NULL)
            throw std::logic_error("tensor component does not exist");
        return *tensors[idx].tensor;
    }

    /**********************************************************************
         *
         * Implementation of Tensor stubs
         *
         *********************************************************************/
    void mult(const T alpha)
    {
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1)
            {
                *tensors[i].tensor *= alpha;
            }
        }
    }

    void mult(const T alpha, const Derived& A,
              const Derived& B, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (tensors.size() != A.tensors.size() ||
                tensors.size() != B.tensors.size()) throw LengthMismatchError();
#endif //VALIDATE_INPUTS

        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1 && A.exists(i) && B.exists(i))
            {
                beta*(*tensors[i].tensor) += alpha*A(i)*B(i);
            }
        }
    }

    void div(const T alpha, const Derived& A,
             const Derived& B, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (tensors.size() != A.tensors.size() ||
                tensors.size() != B.tensors.size()) throw LengthMismatchError();
#endif //VALIDATE_INPUTS

        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1 && A.exists(i) && B.exists(i))
            {
                beta*(*tensors[i].tensor) += alpha*A(i)/B(i);
            }
        }
    }

    void sum(const T alpha, const T beta)
    {
        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1)
            {
                beta*(*tensors[i].tensor) += alpha;
            }
        }
    }

    void sum(const T alpha, const Derived& A, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (tensors.size() != A.tensors.size()) throw LengthMismatchError();
#endif //VALIDATE_INPUTS

        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1 && A.exists(i))
            {
                beta*(*tensors[i].tensor) += alpha*A(i);
            }
        }
    }

    void invert(const T alpha, const Derived& A, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (tensors.size() != A.tensors.size()) throw LengthMismatchError();
#endif //VALIDATE_INPUTS

        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1 && A.exists(i))
            {
                beta*(*tensors[i].tensor) += alpha/A(i);
            }
        }
    }

    T dot(const Derived& A, bool conjb) const
    {
#ifdef VALIDATE_INPUTS
        if (tensors.size() != A.tensors.size()) throw LengthMismatchError();
#endif //VALIDATE_INPUTS

        T s = (T)0;

        for (int i = 0;i < tensors.size();i++)
        {
            if (tensors[i] != NULL && tensors[i].ref == -1 && A.exists(i))
            {
                s += tensors[i].tensor->dot(A(i), conjb);
            }
        }

        return s;
    }
};

template <class Derived, class Base, class T>
class indexable_composite_tensor : public indexable_tensor_base<Derived,T>, public composite_tensor<Derived,Base,T>
{
    INHERIT_FROM_TENSOR(Derived,T)

    protected:
        using indexable_tensor_base<Derived,T>::ndim;

public:
    using indexable_tensor_base< Derived, T >::operator=;
    using indexable_tensor_base< Derived, T >::operator+=;
    using indexable_tensor_base< Derived, T >::operator-=;
    //using composite_tensor<Derived,Base,T>::div;
    //using composite_tensor<Derived,Base,T>::invert;
    using indexable_tensor_base<Derived,T>::scale;
    using indexable_tensor_base<Derived,T>::dot;
    using indexable_tensor_base<Derived,T>::mult;
    using indexable_tensor_base<Derived,T>::sum;
    using indexable_tensor_base<Derived,T>::implicit;

public:
    indexable_composite_tensor(const Derived& other)
        : indexable_tensor_base<Derived,T>(other), composite_tensor<Derived,Base,T>(other) {}

    indexable_composite_tensor(const std::string& name, const Derived& other)
        : indexable_tensor_base<Derived,T>(other), composite_tensor<Derived,Base,T>(name, other) {}

    indexable_composite_tensor(const std::string& name, int ndim=0, int ntensors=0)

        : indexable_tensor_base<Derived,T>(ndim), composite_tensor<Derived,Base,T>(name, ntensors) {}

    virtual ~indexable_composite_tensor() {}

    void mult(const T alpha)
    {
        scale(alpha);
    }

    void mult(const T alpha, const Derived& A,
              const Derived& B,
              const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.ndim || ndim != B_.ndim) throw InvalidNdimError();
#endif //VALIDATE_INPUTS

        mult(alpha, A, A.implicit(),
             B, B.implicit(),
             beta,      implicit());
    }

    virtual Derived& scalar() const = 0;

    void sum(const T alpha, const T beta)
    {
        composite_tensor<Derived,Base,T>::sum(alpha, beta);
    }

    void sum(const T alpha, const Derived& A, const T beta)
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.ndim) throw InvalidNdimError();
#endif //VALIDATE_INPUTS

        sum(alpha, A, A.implicit(),
            beta,      implicit());
    }

    void div(const T alpha, const Derived& A,
             const Derived& B, const T beta)
    {
        composite_tensor<Derived,Base,T>::div(alpha, A, B, beta);
    }

    void invert(const T alpha, const Derived& A, const T beta)
    {
        composite_tensor<Derived,Base,T>::invert(alpha, A, beta);
    }

    void scale(const T alpha)
    {
        scale(alpha, implicit());
    }

    T dot(const Derived& A, bool conjb) const
    {
#ifdef VALIDATE_INPUTS
        if (ndim != A.ndim) throw InvalidNdimError();
#endif //VALIDATE_INPUTS

        return dot(A, A.implicit(),
                   implicit());
    }
};

}
}

#endif
