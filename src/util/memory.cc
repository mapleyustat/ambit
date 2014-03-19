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
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "memory.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>

#include <iostream>

namespace ambit {
namespace util {

void* ambit_malloc(const size_t size_, const char* file, const int line, const int bailout)
{
    assert(ALIGNMENT >= sizeof(size_t));

    size_t size = std::max(size_, size_t(1));

    void *mem;
    if (posix_memalign(&mem, ALIGNMENT, size) != 0) {
        std::cerr
            << " => MEMORY ALLOCATION FAILURE <=\n"
            << "ambit_malloc: Aligned memory allocation failed.\n"
            << "        file: " << file << "\n"
            << "        line: " << line << "\n"
            << "        size: " << size << " (requested: " << size_ << ")\n"
            << "   alignment: " << ALIGNMENT << "\n";
        if (bailout)
            abort();
        return NULL;
    }

    return (void*)((intptr_t)mem);
}

void ambit_free(void* ptr, const char* file, const int line)
{
    if (ptr == NULL) return;
    free(ptr);
}

}

}
