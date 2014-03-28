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

#if !defined(AMBIT_LIB_UTIL_WORLD)
#define AMBIT_LIB_UTIL_WORLD

#if defined(OPENMP_FOUND)
#include <omp.h>
#endif

#if defined(HAVE_MPI)
#include <mpi.h>
#include <ctf.hpp>
#endif // defined(MPI)

#include <boost/shared_ptr.hpp>
#include <vector>

#include "aligned.h"

namespace ambit {
namespace util {

template<typename T>
struct MPI_TYPE_ {};

template<>
struct MPI_TYPE_<char>
{
    static const MPI::Datatype& value() { return MPI::CHAR; }
};

template<>
struct MPI_TYPE_<float>
{
    static const MPI::Datatype& value() { return MPI::FLOAT; }
};

template <>
struct MPI_TYPE_<double>
{
    static const MPI::Datatype& value() { return MPI::DOUBLE; }
};

template <>
struct MPI_TYPE_< std::complex<float> >
{
    static const MPI::Datatype& value() { return MPI::COMPLEX; }
};

template <>
struct MPI_TYPE_< std::complex<double> >
{
    static const MPI::Datatype& value() { return MPI::DOUBLE_COMPLEX; }
};

template <>
struct MPI_TYPE_<short>
{
    static const MPI::Datatype& value() { return MPI::SHORT; }
};

template<>
struct MPI_TYPE_<unsigned short>
{
    static const MPI::Datatype& value() { return MPI::UNSIGNED_SHORT; }
};

template <>
struct MPI_TYPE_<int>
{
    static const MPI::Datatype& value() { return MPI::INT; }
};

template <>
struct MPI_TYPE_<unsigned int>
{
    static const MPI::Datatype& value() { return MPI::UNSIGNED; }
};

template <>
struct MPI_TYPE_<long>
{
    static const MPI::Datatype& value() { return MPI::LONG; }
};

template <>
struct MPI_TYPE_<unsigned long>
{
    static const MPI::Datatype& value() { return MPI::UNSIGNED_LONG; }
};

template <>
struct MPI_TYPE_<long long>
{
    static const MPI::Datatype& value() { return MPI::LONG_LONG; }
};

template <>
struct MPI_TYPE_<unsigned long long>
{
    static const MPI::Datatype& value() { return MPI::UNSIGNED_LONG_LONG; }
};

struct world
{
protected:
    MPI::Intracomm comm;
    boost::shared_ptr<tCTF_World<double> > ctfd;

public:
    const int rank;
    const int nproc;

    World()
        : comm(MPI::COMM_WORLD), rank(comm.Get_rank()), nproc(comm.Get_size()) {}

    World(MPI::Intracomm& comm)
        : comm(comm), rank(comm.Get_rank()), nproc(comm.Get_size()) {}

    void free()
    {
        comm.Free();
    }

    template <typename T>
    tCTF_World<T>& ctf();

    template <typename T>
    const tCTF_World<T>& ctf() const
    {
        return const_cast<const tCTF_World<T>&>(const_cast<World&>(*this).ctf<T>());
    }

    template<typename T>
    void bcast(T* buffer, int count, int root) const
    {
        const MPI::Datatype& type = MPI_TYPE_<T>::value();
        comm.Bcast(buffer, count, type, root);
    }

    template<typename T>
    void bcast(T val, int root) const
    {
        const MPI::Datatype& type = MPI_TYPE_<T>::value();
        comm.Bcast(&val, 1, type, root);
    }

    template<typename T>
    void bcast(std::vector<aligned_vector<T>>& buffer, int root) const
    {
        size_t len=0;
        if (root == rank)
            size_t len = buffer.length();
        bcast(&len, 1, root);

        for (size_t i=0; i<len; ++i) {
            size_t veclen = 0;
            if (root == rank)
                veclen = buffer[i].length();
            bcast(&veclen, 1, root);

            if (root == rank) {
                bcast(buffer[i].data(), veclen, root);
            }
            else {
                aligned_vector<T> avec(veclen);
                bcast(avec.data(), veclen, root);
                buffer.push_back(avec);
            }
        }
    }

    template<typename T>
    void bcast(std::vector<T>& buffer, int root) const
    {
        const MPI::Datatype& type = MPI_TYPE_<T>::value();
        comm.Bcast(buffer.data(), buffer.size(), type, root);
    }

    void bcast(std::string& buffer, int root) const
    {
        int len = buffer.length();
        bcast(len, root);

        buffer.resize(len);
        comm.Bcast((void*)&(buffer[0]), len, MPI_TYPE_<char>::value(), root);
    }
};

template <>
inline tCTF_World<double>& World::ctf<double>()
{
    if (!ctfd) ctfd.reset(new tCTF_World<double>(comm));
    return *ctfd;
}

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


}
}

#endif

