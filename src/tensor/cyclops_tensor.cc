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

#include "cyclops_tensor.h"
#include "util.h"

#include <cfloat>

namespace ambit { namespace tensor {

template<typename T>
CyclopsTensor<T>::CyclopsTensor(const std::string& name, util::World& arena, T scalar)
    : IndexableTensor<CyclopsTensor<T>, T>(name), world(arena), len(0), sym(0)
{
    allocate();
    *dt = scalar;
}

template<typename T>
CyclopsTensor<T>::CyclopsTensor(const std::string& name, const CyclopsTensor<T>& A, T scalar)
    : IndexableTensor<CyclopsTensor<T>, T>(name), world(A.world), len(0), sym(0)
{
    allocate();
    *dt = scalar;
}

template<typename T>
CyclopsTensor<T>::CyclopsTensor(const CyclopsTensor<T>& A, bool copy, bool zero)
    : IndexableTensor<CyclopsTensor<T>, T>(A.name, A.ndim), world(A.world), len(A.len), sym(A.sym)
{
    allocate();

    if (copy)
        *this = A;
    else if (zero)
        *dt = (T)0;
}

template <typename T>
CyclopsTensor<T>::CyclopsTensor(const std::string& name, util::World& arena, const std::vector<int> &len, const std::vector<int> &sym, bool zero)
    : IndexableTensor<CyclopsTensor<T>, T>(name, len.size()), world(arena), len(len), sym(sym)
{
    assert(len.size() == sym.size());

    allocate();
    if (zero)
        *dt = (T)0;
}

template<typename T>
CyclopsTensor<T>::~CyclopsTensor()
{
    free();
}

template<typename T>
void CyclopsTensor<T>::allocate()
{
    dt = new tCTF_Tensor<T>(ndim, len.data(), sym.data(), world.ctf<T>(), "NAME", 1);
}

template<typename T>
void CyclopsTensor<T>::free()
{
    delete dt;
}

template<typename T>
void CyclopsTensor<T>::resize(int _ndim, const std::vector<int> &_len, const std::vector<int> &_sym, bool zero)
{
    assert(_len.size() == ndim);
    assert(_sym.size() == ndim);

    ndim = _ndim;
    len = _len;
    sym = _sym;

    free();
    allocate();
    if (zero)
        *dt = (T)0;
}

template<typename T>
T* CyclopsTensor<T>::get_raw_data(int64_t& size)
{
    return const_cast<T*>(const_cast<const CyclopsTensor<T>&>(*this).get_raw_data(size));
}

template<typename T>
const T* CyclopsTensor<T>::get_raw_data(int64_t& size) const
{
    long_int size_;
    T* data = dt->get_raw_data(&size_);
    size = size_;
    return data;
}

template<typename T>
void CyclopsTensor<T>::read_local(std::vector<tkv_pair<T> >& pairs) const
{
    int64_t npair;
    tkv_pair<T> *data;
    dt->read_local(&npair, &data);
    pairs.assign(data, data+npair);
    if (npair > 0)
        ::free(data);
}

template<typename T>
std::vector<tkv_pair<T> > CyclopsTensor<T>::read_local() const
{
    std::vector<tkv_pair<T> > pairs;
    int64_t npair;
    tkv_pair<T> *data;
    dt->read_local(&npair, &data);
    pairs.assign(data, data+npair);
    if (npair > 0)
        ::free(data);
    return pairs;
}

template<typename T>
void CyclopsTensor<T>::read(std::vector<tkv_pair<T> >& pairs) const
{
    dt->read(pairs.size(), pairs.data());
}

template<typename T>
void CyclopsTensor<T>::read() const
{
    dt->read(0, NULL);
}

template<typename T>
void CyclopsTensor<T>::write(const std::vector<tkv_pair<T> >& pairs)
{
    dt->write(pairs.size(), pairs.data());
}

template<typename T>
void CyclopsTensor<T>::write()
{
    dt->write(0, NULL);
}

//template <typename T>
//void CyclopsTensor<T>::write_remote_data(double alpha, double beta, const std::vector<tkv_pair<T> >& pairs)
//{
//    dt->add_remote_data(pairs.size(), alpha, beta, pairs.data());
//}

//template <typename T>
//void CyclopsTensor<T>::write_remote_data(double alpha, double beta)
//{
//    dt->add_remote_data(0, alpha, beta, NULL);
//}

template <typename T>
void CyclopsTensor<T>::get_all_data(std::vector<T>& vals) const
{
    get_all_data(vals, 0);
    int64_t npair = vals.size();

    world.bcast(&npair, 1, 0);
    if (world.rank != 0) vals.resize(npair);

    world.bcast(vals, 0);
}

template <typename T>
void CyclopsTensor<T>::get_all_data(std::vector<T>& vals, int rank) const
{
    if (world.rank == rank)
    {
        std::vector<tkv_pair<T> > pairs;
        std::vector<int> idx(ndim, 0);

        first_packed_indices(ndim, len.data(), sym.data(), idx.data());

        do {
            int64_t key = 0, stride = 1;
            for (int i = 0;i < ndim;i++) {
                key += idx[i]*stride;
                stride *= len[i];
            }
            pairs.push_back(tkv_pair<T>(key, (T)0));
        }
        while (next_packed_indices(ndim, len.data(), sym.data(), idx.data()));

        dt->read(pairs.size(), pairs.data());

        std::sort(pairs.begin(), pairs.end());
        size_t npair = pairs.size();
        vals.resize(npair);

        for (size_t i = 0;i < npair;i++)
            vals[i] = pairs[i].d;
    }
    else {
        dt->read(0, NULL);
    }
}

template <typename T>
void CyclopsTensor<T>::div(T alpha, const CyclopsTensor<T>& A,
                                        const CyclopsTensor<T>& B, T beta)
{
    const_cast<tCTF_Tensor<T>*>(A.dt)->align(*dt);
    const_cast<tCTF_Tensor<T>*>(B.dt)->align(*dt);
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
void CyclopsTensor<T>::invert(T alpha, const CyclopsTensor<T>& A, T beta)
{
    dt->align(*A.dt);
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
void CyclopsTensor<T>::print() const
{
    dt->print(stdout, 1.0e-10);
}

template <typename T>
void CyclopsTensor<T>::compare(const CyclopsTensor<T>& other, double cutoff) const
{
    dt->compare(*other.dt, stdout, cutoff);
}

template <typename T>
typename real_type<T>::type CyclopsTensor<T>::norm(int p) const
{
    T ans = (T)0;
    if (p == 0)
        ans = dt->reduce(CTF_OP_NORM_INFTY);
    else if (p == 1)
        ans = dt->reduce(CTF_OP_NORM1);
    else if (p == 2)
        ans = dt->reduce(CTF_OP_NORM2);
    return abs(ans);
}

template <typename T>
void CyclopsTensor<T>::mult(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A,
                                     const CyclopsTensor<T>& B, const std::string& idx_B,
                            T  beta,                            const std::string& idx_C)
{
    dt->contract(alpha, *A.dt, idx_A.c_str(),
                        *B.dt, idx_B.c_str(),
                  beta,        idx_C.c_str());
}

template <typename T>
void CyclopsTensor<T>::sum(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A,
                           T  beta,                            const std::string& idx_B)
{
    dt->sum(alpha, *A.dt, idx_A.c_str(),
             beta,        idx_B.c_str());
}

template <typename T>
void CyclopsTensor<T>::scale(T alpha, const std::string& idx_A)
{
    dt->scale(alpha, idx_A.c_str());
}

template <typename T>
T CyclopsTensor<T>::dot(const CyclopsTensor<T>& A, const std::string& idx_A,
                                                   const std::string& idx_B) const
{
    CyclopsTensor<T> dt(A.name, A.world);
    std::vector<T> val;
    dt.mult(1,     A, idx_A,
               *this, idx_B,
            0,           "");
    dt.get_all_data(val);
    assert(val.size()==1);
    return val[0];
}

template <typename T>
void CyclopsTensor<T>::weight(const std::vector<const std::vector<T>*>& d)
{
    assert(d.size() == ndim);
    for (int i = 0;i < d.size();i++) assert(d[i]->size() == len[i]);

    std::vector<tkv_pair<T> > pairs;
    read_local(pairs);

    for (int i = 0; i < pairs.size(); ++i) {
        int64_t k = pairs[i].k;

        T den = 0;
        for (int j = 0; j < ndim; ++j) {
            int o = k%len[j];
            k = k/len[j];
            den += (*d[j])[o];
        }

        pairs[i].d /= den;
    }

    write(pairs);
}

template <typename T>
void CyclopsTensor<T>::fill_with_random_data()
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
void CyclopsTensor<T>::sort(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A, const std::string& idx_B)
{
    (*dt)[idx_B.c_str()] = alpha * (*A.dt)[idx_A.c_str()];
}

INSTANTIATE_SPECIALIZATIONS(CyclopsTensor)

}}
