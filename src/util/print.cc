/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
 */

#include "print.h"

#include <cstdio>
#include <cstdarg>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace ambit { namespace util {

void print0(const std::string format, ...)
{
#if defined(HAVE_MPI)
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
#endif
        va_list args;
        va_start(args, format);
        vprintf(format.c_str(), args);
        va_end(args);
#if defined(HAVE_MPI)
    }
#endif
}

}}

