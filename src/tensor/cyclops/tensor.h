/*
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
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#if !defined(AMBIT_LIB_TENSOR_CYCLOPS_TENSOR)
#define AMBIT_LIB_TENSOR_CYCLOPS_TENSOR

#if !defined(HAVE_MPI)
#   error MPI is not defined and is required for compiling Cyclops tensor.
#endif

#include <ctf.hpp>
#include "../tensor.h"
#include "world.h"

#include <vector>

namespace ambit { namespace tensor { namespace cyclops {

template <typename T>
struct tensor : public indexable_tensor< tensor<T>, T>
{
    using tensor_base< tensor<T>, T>::ndim_;
    using indexable_tensor< tensor<T>, T>::ir_;

protected:
    tCTF_Tensor<T> *dt_;

    // Cyclops specific way of loading local data.
    void read_local(std::vector<tkv_pair<T> >& pairs) const
    {
        int64_t npair = 0;
        tkv_pair<T> *data;
        dt_->read_local(&npair, &data);
        pairs.assign(data, data+npair);
        if (npair > 0)
            ::free(data);
    }

public:
    void write(const std::vector<tkv_pair<T> >& pairs)
    {
        dt_->write(pairs.size(), pairs.data());
    }

    void write()
    {
        dt_->write(0, NULL);
    }

    using indexable_tensor<tensor<T>, T>::operator=;
    //tensor& operator=(const tensor& other) = delete;

    tensor(const std::string& name, const std::string& indices) :
        indexable_tensor<tensor<T>, T>(name, indices)
    {
        std::vector<int> len(ndim_), sym(ndim_);

        for (int i=0; i<ndim_; ++i) {
            len[i] = ir_[i].length();
            sym[i] = 0;
        }
        dt_ = new tCTF_Tensor<T>(ndim_, len.data(), sym.data(), world::shared().ctf<T>(), name.c_str(), 1);
    }

    virtual ~tensor()
    {
        delete dt_;
    }

    void print() const
    {
        dt_->print(stdout, 1.0e-10);
    }

    void fill_random()
    {
        std::vector<tkv_pair<T> > pairs;

        // get our local pairs
        read_local(pairs);

        // fill with random data
        for (auto& a : pairs)
            a.d = drand48() - 0.5;

        // save our local data to the tensor
        write(pairs);
    }

    // this = alpha * this + beta * A * B
    // this = alpha * this + beta * A * B
    void multiply(const T& alpha, const indexed_tensor<tensor, T>& A,
                                  const indexed_tensor<tensor, T>& B,
                  const T& beta,  const std::string& index_C)
    {
        dt_->contract(alpha, *A.tensor_.dt_, A.index_.c_str(),
                             *B.tensor_.dt_, B.index_.c_str(),
                      beta,                  index_C.c_str());
    }

    void sum(const T& alpha, const indexed_tensor<tensor, T>& A,
             const T& beta,  const std::string& index_B)
    {
        dt_->sum(alpha, *A.tensor_.dt_, A.index_.c_str(),
                 beta,                  index_B.c_str());
    }
};

} // namespace cyclops

using tensor = cyclops::tensor<double>;

}}

#endif
