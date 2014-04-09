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

#if !defined(AMBIT_SRC_UTIL_PRINT)
#define AMBIT_SRC_UTIL_PRINT

#include <string>

namespace ambit { namespace util {

/// Only the master process is allowed to print to the screen.
void print0(const std::string format, ...);

namespace print {

/** Initialize the printing system.
 *
 * Will open a file for each MPI process. Only that process will be allowed to print to that
 * file.
 */
bool initialize();

/** Closes any open file handles in the printing system.
 */
void finalize();

/** Print a nice banner to the printn output stream. Do not include \n at the end of the line.
 */
void banner(const std::string format, ...);

/** Increases printing offset by increment.
 * \param increment the amount to increase indentation.
 */
void indent(int increment = 4);

/** Decreases printing offset by increment.
 * \param decrement the amount to decrease indentation.
 */
void unindent(int decrement = 4);

struct indenter
{
    indenter(int increment = 4) : size(increment) { indent(size); }
    ~indenter() { unindent(size); }

private:
    int size;
};

} // namespace print

/** Each process will print to their respective output file.
 * The master process will print to both the screen and its output file.
 */
void printn(const std::string format, ...);

}}

#endif

