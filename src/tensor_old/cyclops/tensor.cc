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

#include "tensor.h"
#include "../indices.h"
#include "../util.h"

#include <cfloat>

namespace ambit { namespace tensor { namespace cyclops {

template<typename T>
tensor<T>::tensor(const std::string& name, world& arena, T scalar)
    : indexable_tensor<tensor<T>, T>(name), world_(arena), len_(0), sym_(0)
{
    allocate();
    *dt_ = scalar;
}

template<typename T>
tensor<T>::tensor(const std::string& name, const tensor<T>& A, T scalar)
    : indexable_tensor<tensor<T>, T>(name), world_(A.world_), len_(0), sym_(0)
{
    allocate();
    *dt_ = scalar;
}

template<typename T>
tensor<T>::tensor(const tensor<T>& A, bool copy, bool zero)
    : indexable_tensor<tensor<T>, T>(A.name, A.ndim), world_(A.world_), len_(A.len_), sym_(A.sym_)
{
    allocate();

    if (copy)
        *this = A;
    else if (zero)
        *dt_ = (T)0;
}

template <typename T>
tensor<T>::tensor(const std::string& name, world& arena, const std::vector<int> &len, const std::vector<int> &sym, bool zero)
    : indexable_tensor<tensor<T>, T>(name, len.size()), world_(arena), len_(len), sym_(sym)
{
    assert(len.size() == sym.size());

    allocate();
    if (zero)
        *dt_ = (T)0;
}

template<typename T>
tensor<T>::~tensor()
{
    free();
}

template<typename T>
void tensor<T>::allocate()
{
    dt_ = new tCTF_Tensor<T>(ndim, len_.data(), sym_.data(), world_.ctf<T>(), "NAME", 1);
}

template<typename T>
void tensor<T>::free()
{
    delete dt_;
}

template<typename T>
void tensor<T>::resize(int _ndim, const std::vector<int> &_len, const std::vector<int> &_sym, bool zero)
{
    assert(_len.size() == ndim);
    assert(_sym.size() == ndim);

    ndim = _ndim;
    len_ = _len;
    sym_ = _sym;

    free();
    allocate();
    if (zero)
        *dt_ = (T)0;
}

template<typename T>
T* tensor<T>::get_raw_data(int64_t& size)
{
    return const_cast<T*>(const_cast<const tensor<T>&>(*this).get_raw_data(size));
}

template<typename T>
const T* tensor<T>::get_raw_data(int64_t& size) const
{
    long_int size_;
    T* data = dt_->get_raw_data(&size_);
    size = size_;
    return data;
}

template<typename T>
void tensor<T>::read_local(std::vector<tkv_pair<T> >& pairs) const
{
    int64_t npair;
    tkv_pair<T> *data;
    dt_->read_local(&npair, &data);
    pairs.assign(data, data+npair);
    if (npair > 0)
        ::free(data);
}

template<typename T>
std::vector<tkv_pair<T> > tensor<T>::read_local() const
{
    std::vector<tkv_pair<T> > pairs;
    int64_t npair;
    tkv_pair<T> *data;
    dt_->read_local(&npair, &data);
    pairs.assign(data, data+npair);
    if (npair > 0)
        ::free(data);
    return pairs;
}

template<typename T>
void tensor<T>::read(std::vector<tkv_pair<T> >& pairs) const
{
    dt_->read(pairs.size(), pairs.data());
}

template<typename T>
void tensor<T>::read() const
{
    dt_->read(0, NULL);
}

template<typename T>
void tensor<T>::write(const std::vector<tkv_pair<T> >& pairs)
{
    dt_->write(pairs.size(), pairs.data());
}

template<typename T>
void tensor<T>::write()
{
    dt_->write(0, NULL);
}

//template <typename T>
//void tensor<T>::write_remote_data(double alpha, double beta, const std::vector<tkv_pair<T> >& pairs)
//{
//    dt_->add_remote_data(pairs.size(), alpha, beta, pairs.data());
//}

//template <typename T>
//void tensor<T>::write_remote_data(double alpha, double beta)
//{
//    dt_->add_remote_data(0, alpha, beta, NULL);
//}

template <typename T>
void tensor<T>::get_all_data(std::vector<T>& vals) const
{
    get_all_data(vals, 0);
    int64_t npair = vals.size();

    world_.bcast(&npair, 1, 0);
    if (world_.rank != 0) vals.resize(npair);

    world_.bcast(vals, 0);
}

template <typename T>
void tensor<T>::get_all_data(std::vector<T>& vals, int rank) const
{
    if (world_.rank == rank)
    {
        std::vector<tkv_pair<T> > pairs;
        std::vector<int> idx(ndim, 0);

        first_packed_indices(ndim, len_.data(), sym_.data(), idx.data());

        do {
            int64_t key = 0, stride = 1;
            for (int i = 0;i < ndim;i++) {
                key += idx[i]*stride;
                stride *= len_[i];
            }
            pairs.push_back(tkv_pair<T>(key, (T)0));
        }
        while (next_packed_indices(ndim, len_.data(), sym_.data(), idx.data()));

        dt_->read(pairs.size(), pairs.data());

        std::sort(pairs.begin(), pairs.end());
        size_t npair = pairs.size();
        vals.resize(npair);

        for (size_t i = 0;i < npair;i++)
            vals[i] = pairs[i].d;
    }
    else {
        dt_->read(0, NULL);
    }
}

template <typename T>
void tensor<T>::div(T alpha, const tensor<T>& A,
                             const tensor<T>& B, T beta)
{
    const_cast<tCTF_Tensor<T>*>(A.dt_)->align(*dt_);
    const_cast<tCTF_Tensor<T>*>(B.dt_)->align(*dt_);
    int64_t size, size_A, size_B;
    T* raw_data = get_raw_data(size);
    const T* raw_data_A = A.get_raw_data(size_A);
    const T* raw_data_B = B.get_raw_data(size_B);
    assert(size == size_A);
    assert(size == size_B);

    for (int64_t i = 0;i < size;i++) {
        if (abs(raw_data_B[i]) > DBL_MIN)
            raw_data[i] = beta*raw_data[i] + alpha*raw_data_A[i]/raw_data_B[i];
    }
}

template <typename T>
void tensor<T>::invert(T alpha, const tensor<T>& A, T beta)
{
    dt_->align(*A.dt_);
    int64_t size, size_A;
    T* raw_data = get_raw_data(size);
    const T* raw_data_A = A.get_raw_data(size_A);
    assert(size == size_A);
    for (int64_t i = 0; i < size; ++i) {
        if (abs(raw_data_A[i]) > DBL_MIN)
            raw_data[i] = beta*raw_data[i] + alpha/raw_data_A[i];
    }
}

template <typename T>
void tensor<T>::print() const
{
    dt_->print(stdout, 1.0e-10);
}

template <typename T>
void tensor<T>::compare(const tensor<T>& other, double cutoff) const
{
    dt_->compare(*other.dt_, stdout, cutoff);
}

template <typename T>
typename real_type<T>::type tensor<T>::norm(int p) const
{
    T ans = (T)0;
    if (p == 0)
        ans = dt_->reduce(CTF_OP_NORM_INFTY);
    else if (p == 1)
        ans = dt_->reduce(CTF_OP_NORM1);
    else if (p == 2)
        ans = dt_->reduce(CTF_OP_NORM2);
    return abs(ans);
}

template <typename T>
void tensor<T>::mult(T alpha, const tensor<T>& A, const std::string& idx_A,
                                     const tensor<T>& B, const std::string& idx_B,
                            T  beta,                            const std::string& idx_C)
{
    dt_->contract(alpha, *A.dt_, idx_A.c_str(),
                         *B.dt_, idx_B.c_str(),
                  beta,          idx_C.c_str());
}

template <typename T>
void tensor<T>::sum(T alpha, const tensor<T>& A, const std::string& idx_A,
                           T  beta,                            const std::string& idx_B)
{
    dt_->sum(alpha, *A.dt_, idx_A.c_str(),
             beta,          idx_B.c_str());
}

template <typename T>
void tensor<T>::scale(T alpha, const std::string& idx_A)
{
    dt_->scale(alpha, idx_A.c_str());
}

template <typename T>
T tensor<T>::dot(const tensor<T>& A, const std::string& idx_A,
                                                   const std::string& idx_B) const
{
    tensor<T> dt(A.name, A.world_);
    std::vector<T> val;
    dt.mult(1,     A, idx_A,
               *this, idx_B,
            0,           "");
    dt.get_all_data(val);
    assert(val.size()==1);
    return val[0];
}

template <typename T>
void tensor<T>::weight(const std::vector<const std::vector<T>*>& d)
{
    assert(d.size() == ndim);
    for (int i = 0;i < d.size();i++) assert(d[i]->size() == len_[i]);

    std::vector<tkv_pair<T> > pairs;
    read_local(pairs);

    for (int i = 0; i < pairs.size(); ++i) {
        int64_t k = pairs[i].k;

        T den = 0;
        for (int j = 0; j < ndim; ++j) {
            int o = k%len_[j];
            k = k/len_[j];
            den += (*d[j])[o];
        }

        pairs[i].d /= den;
    }

    write(pairs);
}

template <typename T>
void tensor<T>::fill_with_random_data()
{
    std::vector<tkv_pair<T> > pairs;

    // Get our local pairs
    read_local(pairs);

    // Set random values for only our data
    for (int i=0; i<pairs.size(); ++i) {
        pairs[i].d = drand48()-.5;
    }

    // Save our local data to the CTF tensor.
    write(pairs);
}

template <typename T>
void tensor<T>::sort(T alpha, const tensor<T>& A, const std::string& idx_A, const std::string& idx_B)
{
    (*dt_)[idx_B.c_str()] = alpha * (*A.dt_)[idx_A.c_str()];
}

INSTANTIATE_SPECIALIZATIONS(tensor)

}}}
