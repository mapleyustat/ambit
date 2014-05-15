/*
 * Copyright (C) 2014  Justin Turney
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

#if !defined(AMBIT_TENSOR_EXCEPTION_H)
#define AMBIT_TENSOR_EXCEPTION_H

#include <stdexcept>

namespace ambit { namespace tensor {

struct tensor_error : public std::exception
{
    virtual const char * what() const throw() = 0;
};

struct index_not_found_error : public tensor_error
{
    virtual const char * what() const throw() {
        return "index not found";
    }
};

struct index_already_exists_error : public tensor_error
{
    virtual const char * what() const throw() {
        return "index already exists";
    }
};

struct invalid_ndim_error : public std::exception
{
    virtual const char * what() const throw() {
        return "invalid ndim";
    }
};

}}

#endif
