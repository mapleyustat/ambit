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

#include <boost/lexical_cast.hpp>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace ambit { namespace util {

namespace print {

namespace {
FILE *process_file_handle = nullptr;
int indent_size = 0;

void print_indentation(FILE *out)
{
    fprintf(out, "%*s", indent_size, "");
}

void banner()
{
    fprintf(process_file_handle,
            "********************************************************************\n"
            "*                          New run started.                        *\n"
            "********************************************************************\n");
}

int get_comm_rank()
{
    int rank = 0;
#if defined(HAVE_MPI)
    int flag;
    MPI_Initialized(&flag);
    if (flag)
        rank = MPI::COMM_WORLD.Get_rank();
#endif
    return rank;
}

} // namespace anonymous

void indent(int increment)
{
    indent_size += increment;
}

void unindent(int decrement)
{
    indent_size -= decrement;
    if (indent_size < 0)
        indent_size = 0;
}

bool initialize()
{
    int rank = 0;
#if defined(HAVE_MPI)
    int flag = 0;
    MPI_Initialized(&flag);

    if (flag)
        rank = MPI::COMM_WORLD.Get_rank();
    else
        rank = 0;
#endif

    std::string srank = boost::lexical_cast<std::string>(rank) + ".pout";

    process_file_handle = fopen(srank.c_str(), "a");

    if (process_file_handle == NULL)
        return false;

    // turn off buffering
    setvbuf(process_file_handle, NULL, _IONBF, 0);

    banner();

    return true;
}

void finalize()
{
    fclose(process_file_handle);
    process_file_handle = nullptr;
}

void banner(const std::string format, ...)
{
    fprintf(process_file_handle, "%*s ==> ", indent_size, "");
    va_list args;
    va_start(args, format);
    vfprintf(process_file_handle, format.c_str(), args);
    va_end(args);
    fprintf(process_file_handle, " <==\n");

    if (print::get_comm_rank() == 0) {
        printf("%*s ==> ", indent_size, "");
        va_list args;
        va_start(args, format);
        vprintf(format.c_str(), args);
        va_end(args);
        printf(" <==\n");
    }
}

}

void print0(const std::string format, ...)
{
#if defined(HAVE_MPI)
    int rank = 0, flag;
    MPI_Initialized(&flag);
    if (flag)
        rank = MPI::COMM_WORLD.Get_rank();

    if (rank == 0) {
#endif
        va_list args;
        va_start(args, format);
        print::print_indentation(stdout);
        vprintf(format.c_str(), args);
        va_end(args);
#if defined(HAVE_MPI)
    }
#endif
}

void printn(const std::string format, ...)
{
    va_list args;
    va_start(args, format);
    print::print_indentation(print::process_file_handle);
    vfprintf(print::process_file_handle, format.c_str(), args);
    va_end(args);
    fflush(print::process_file_handle);

    static int rank = print::get_comm_rank();
    if (rank == 0) {
        print::print_indentation(stdout);
        va_start(args, format);
        vprintf(format.c_str(), args);
        va_end(args);
    }
}

}}

