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

#if !defined(MINTS_LIB_UTIL_TIMER)
#define MINTS_LIB_UTIL_TIMER

#include <cstddef>

#define SAFE_MALLOC(type, size) (type*)ambit::util::ambit_malloc(sizeof(type)*(size), __FILE__, __LINE__, 1)
#define FREE(ptr) ambit::util::ambit_free(ptr, __FILE__, __LINE__)

namespace ambit {
namespace util {

// Recommended alignment value from Intel
enum { ALIGNMENT = 64 };

#ifdef __INTEL_COMPILER

#define ALIGNED_LOOP(x) \
_Pragma("ivdep") \
_Pragma("vector aligned") \
for (x)

#else

#define ALIGNED_LOOP(x) \
for (x)

#endif

extern size_t mem_used;

void* ambit_malloc(const size_t size, const char* who, const int where, const int bailout);
void ambit_free(void* ptr, const char* who, const int where);

}
}

#endif

