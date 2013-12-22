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

#if !defined(MPI)
#   error MPI is not defined and is required for compiling CyclopsTensor.
#endif

#include <ctf.hpp>
#include "indexable_tensor.h"
#include <util/world.h>

#include <vector>

namespace ambit {

namespace tensor {

template <typename T>
struct CyclopsTensor : public IndexableTensor< CyclopsTensor<T>, T>
{
    INHERIT_FROM_INDEXABLE_TENSOR(CyclopsTensor<T>,T)

protected:
    tCTF_Tensor<T> *dt;
    util::World& world;
    std::vector<int> len;
    std::vector<int> sym;

    void allocate();
    void free();

public:
    CyclopsTensor(const std::string& name, util::World& arena, T scalar = (T)0);
    CyclopsTensor(const std::string& name, const CyclopsTensor<T>& A, T scalar);

    CyclopsTensor(const CyclopsTensor<T>& A, bool copy=true, bool zero=false);

    CyclopsTensor(const std::string& name, util::World& arena, const std::vector<int>& len, const std::vector<int>& sym,
                bool zero=true);

    ~CyclopsTensor();

    void resize(int ndim, const std::vector<int>& len, const std::vector<int>& sym, bool zero);

    const std::vector<int>& get_lengths() const { return len; }
    const std::vector<int>& get_symmetry() const { return sym; }

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

    void div(T alpha, const CyclopsTensor<T>& A,
                      const CyclopsTensor<T>& B, T beta);

    void invert(T alpha, const CyclopsTensor<T>& A, T beta);

    void weight(const std::vector<const std::vector<T>*>& d);

    void print() const;

    void compare(const CyclopsTensor<T>& other, double cutoff = 0.0) const;

    typename real_type<T>::type norm(int p) const;

    void mult(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A,
                       const CyclopsTensor<T>& B, const std::string& idx_B,
              T  beta,                                const std::string& idx_C);

    void sum(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A,
             T beta,                                 const std::string& idx_B);

    void scale(T alpha, const std::string& idx_A);

    T dot(const CyclopsTensor<T>& A, const std::string& idx_A,
                                         const std::string& idx_B) const;

    /// Performs this[idx_B] = factor * A[idx_A]
    void sort(T alpha, const CyclopsTensor<T>& A, const std::string& idx_A, const std::string& idx_B);
};

}

}

#endif
