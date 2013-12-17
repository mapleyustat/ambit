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

#include <string>
#include <chrono>

namespace mints {
namespace util {

struct timer
{
    typedef std::chrono::high_resolution_clock clock;

    timer(std::string const & name);
    ~timer();

    clock::duration time_elapsed() const { return clock::now() - epoch_; }

private:
    std::string const name_;
    clock::time_point epoch_;
};

}
}

#endif

