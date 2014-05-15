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

#if !defined(AMBIT_LIB_TENSOR_UTIL)
#define AMBIT_LIB_TENSOR_UTIL

#include <vector>
#include <algorithm>

namespace ambit {

template <bool cond, class return_type = void> struct enable_if {};
template <class return_type> struct enable_if<true, return_type> { typedef return_type type; };
template <> struct enable_if<true> { typedef void type; };

template <class T, class U> struct is_same      { static const bool value = false; };
template <class T>          struct is_same<T,T> { static const bool value = true; };
template <class T> struct remove_cv                   {typedef T type; };
template <class T> struct remove_cv<const T>          {typedef T type; };
template <class T> struct remove_cv<volatile T>       {typedef T type; };
template <class T> struct remove_cv<const volatile T> {typedef T type; };
template <class T> struct make_unsigned__                   { typedef T type; };
template <>        struct make_unsigned__<signed char>      { typedef unsigned char type; };
template <>        struct make_unsigned__<signed short>     { typedef unsigned short type; };
template <>        struct make_unsigned__<signed int>       { typedef unsigned int type; };
template <>        struct make_unsigned__<signed long>      { typedef unsigned long type; };
template <>        struct make_unsigned__<signed long long> { typedef unsigned long long type; };
template <class T> struct make_unsigned                   { typedef typename make_unsigned__<T>::type type; };
template <class T> struct make_unsigned<const T>          { typedef const typename make_unsigned__<T>::type type; };
template <class T> struct make_unsigned<volatile T>       { typedef volatile typename make_unsigned__<T>::type type; };
template <class T> struct make_unsigned<const volatile T> { typedef const volatile typename make_unsigned__<T>::type type; };
template <class T> struct is_const          { static const bool value = false; };

template <class T> struct is_const<const T> { static const bool value = true; };
template <class T, class U = void> struct is_integral { static const bool value = false; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<unsigned char,typename remove_cv<typename make_unsigned<T>::type>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<unsigned short,typename remove_cv<typename make_unsigned<T>::type>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<unsigned int,typename remove_cv<typename make_unsigned<T>::type>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<unsigned long,typename remove_cv<typename make_unsigned<T>::type>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<unsigned long long,typename remove_cv<typename make_unsigned<T>::type>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<bool,typename remove_cv<T>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_integral<T,
    typename enable_if<is_same<wchar_t,typename remove_cv<T>::type>::value>::type>
    { static const bool value = true; };
template <class T, class U = void> struct is_floating_point { static const bool value = false; };
template <class T> struct is_floating_point<T,
    typename enable_if<is_same<float,typename remove_cv<T>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_floating_point<T,
    typename enable_if<is_same<double,typename remove_cv<T>::type>::value>::type>
    { static const bool value = true; };
template <class T> struct is_floating_point<T,
    typename enable_if<is_same<long double,typename remove_cv<T>::type>::value>::type>
    { static const bool value = true; };

template <class T, class U = void> struct is_arithmetic { static const bool value = false; };
template <class T> struct is_arithmetic<T,
    typename enable_if<is_integral<T>::value>::type>
    { static const bool value = true; };
template <class T> struct is_arithmetic<T,
    typename enable_if<is_floating_point<T>::value>::type>
    { static const bool value = true; };

template<typename T> typename enable_if<!is_arithmetic<T>::value,T>::type sum(const std::vector<T>& v)
{
    T s;
    for (int i = 0;i < v.size();i++) s += v[i];
    return s;
}
template<typename T> typename enable_if<is_arithmetic<T>::value,T>::type sum(const std::vector<T>& v)
{
    T s = (T)0;
    for (int i = 0;i < v.size();i++) s += v[i];
    return s;
}

template<typename T, typename U> bool contains(const T& v, const U& e)
{
    return find(v.begin(), v.end(), e) != v.end();
}

template<typename T> T& uniq(T& v)
{
    typename T::iterator i1;
    std::sort(v.begin(), v.end());
    i1 = std::unique(v.begin(), v.end());
    v.resize(i1-v.begin());
    return v;
}

template<typename T> T& exclude(T& v1, const T& v2)
{
    T v3(v2);
    typename T::iterator i1, i2, i3;
    std::sort(v1.begin(), v1.end());
    std::sort(v3.begin(), v3.end());
    i1 = i2 = v1.begin();
    i3 = v3.begin();
    while (i1 != v1.end())
    {
        if (i3 == v3.end() || *i1 < *i3)
        {
            *i2 = *i1;
            ++i1;
            ++i2;
        }
        else if (*i3 < *i1)
        {
            ++i3;
        }
        else
        {
            ++i1;
        }
    }
    v1.resize(i2-v1.begin());

    return v1;
}

void first_packed_indices(const int ndim, const int* len, const int* sym, int* idx);
bool next_packed_indices(const int ndim, const int* len, const int* sym, int* idx);

}

#endif
