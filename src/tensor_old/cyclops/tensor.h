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

#if !defined(AMBIT_LIB_TENSOR_CYCLOPS_TENSOR)
#define AMBIT_LIB_TENSOR_CYCLOPS_TENSOR

#if !defined(HAVE_MPI)
#   error MPI is not defined and is required for compiling tensor.
#endif

#include <ctf.hpp>
#include "../indexable_tensor.h"
#include "world.h"

#include <vector>

namespace ambit { namespace tensor { namespace cyclops {

template <typename T>
struct tensor : public indexable_tensor< tensor<T>, T>
{
    INHERIT_FROM_INDEXABLE_TENSOR(tensor<T>,T)

protected:
    tCTF_Tensor<T> *dt_;
    world& world_;
    std::vector<int> len_;
    std::vector<int> sym_;

    void allocate();
    void free();

public:
    tensor(const std::string& name, world& arena, T scalar = (T)0);
    tensor(const std::string& name, const tensor<T>& A, T scalar);

    tensor(const tensor<T>& A, bool copy=true, bool zero=false);

    tensor(const std::string& name, world& arena, const std::vector<int>& len, const std::vector<int>& sym,
                bool zero=true);

    ~tensor();

    void resize(int ndim, const std::vector<int>& len, const std::vector<int>& sym, bool zero);

    const std::vector<int>& get_lengths() const { return len_; }
    const std::vector<int>& get_symmetry() const { return sym_; }

    void fill_with_random_data();

    T* get_raw_data(int64_t& size);
    const T* get_raw_data(int64_t& size) const;

    void read_local(std::vector<tkv_pair<T> >& pairs) const;
    std::vector<tkv_pair<T> > read_local() const;
    void read(std::vector<tkv_pair<T> >& pairs) const;
    void read() const;

    void write(const std::vector<tkv_pair<T> >& pairs);
    void write();

//    void write_remote_data(double alpha, double beta, const std::vector<tkv_pair<T> >& pairs);
//    void write_remote_data(double alpha, double beta);

    void get_all_data(std::vector<T>& vals) const;
    void get_all_data(std::vector<T> &vals, int rank) const;

    void div(T alpha, const tensor<T>& A,
                      const tensor<T>& B, T beta);

    void invert(T alpha, const tensor<T>& A, T beta);

    void weight(const std::vector<const std::vector<T>*>& d);

    void print() const;

    void compare(const tensor<T>& other, double cutoff = 0.0) const;

    typename real_type<T>::type norm(int p) const;

    void mult(T alpha, const tensor<T>& A, const std::string& idx_A,
                       const tensor<T>& B, const std::string& idx_B,
              T  beta,                     const std::string& idx_C);

    void sum(T alpha, const tensor<T>& A, const std::string& idx_A,
             T beta,                      const std::string& idx_B);

    void scale(T alpha, const std::string& idx_A);

    T dot(const tensor<T>& A, const std::string& idx_A,
                              const std::string& idx_B) const;

    /// Performs this[idx_B] = factor * A[idx_A]
    void sort(T alpha, const tensor<T>& A, const std::string& idx_A, const std::string& idx_B);
};

} } }

#endif
