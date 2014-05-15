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

#if !defined(AMBIT_LIB_TENSOR_LOCAL_WORLD)
#define AMBIT_LIB_TENSOR_LOCAL_WORLD

#if defined(OPENMP_FOUND)
#include <omp.h>
#endif

#include <vector>

#include <util/aligned.h>

namespace ambit { namespace tensor { namespace local {

struct world
{
    const int rank;
    const int nproc;

    world()
        : rank(0), nproc(1)
    { }

    void free()
    { }

    template<typename T>
    void bcast(T* buffer, int count, int root) const
    { }

    template<typename T>
    void bcast(T& val, int root) const
    { }

    template<typename T>
    void bcast(aligned_vector<T>& buffer, int root) const
    { }

    template<typename T>
    void bcast(std::vector<aligned_vector<T>>& buffer, int root) const
    { }

    template<typename T>
    void bcast(std::vector<T>& buffer, int root) const
    { }

    void bcast(std::string& buffer, int root) const
    { }

    void bcast(std::vector<std::string>& buffer, int root) const
    { }

    static world& shared();
};

// Bad place for this:
template<typename T> std::vector<T> vec(const T& a)
{
    std::vector<T> v;
    v.push_back(a);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e,
                                        const T& f)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    v.push_back(f);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e,
                                        const T& f, const T& g)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    v.push_back(f);
    v.push_back(g);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e,
                                        const T& f, const T& g, const T& h)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    v.push_back(f);
    v.push_back(g);
    v.push_back(h);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e,
                                        const T& f, const T& g, const T& h, const T& i)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    v.push_back(f);
    v.push_back(g);
    v.push_back(h);
    v.push_back(i);
    return v;
}

template<typename T> std::vector<T> vec(const T& a, const T& b, const T& c, const T& d, const T& e,
                                        const T& f, const T& g, const T& h, const T& i, const T& j)
{
    std::vector<T> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
    v.push_back(e);
    v.push_back(f);
    v.push_back(g);
    v.push_back(h);
    v.push_back(i);
    v.push_back(j);
    return v;
}


} } }

#endif

