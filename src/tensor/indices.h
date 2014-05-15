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

#if !defined(AMBIT_TENSOR_INDICES_H)
#define AMBIT_TENSOR_INDICES_H

#include <string>
#include <vector>
#include <map>
#include <numeric>

namespace ambit { namespace tensor {

struct index_range
{
    typedef std::map<std::string, index_range> set_type;
    typedef set_type::iterator set_iterator;
    typedef set_type::const_iterator set_const_iterator;

    static set_type set;

    std::string name;
    int start;
    int end;
    char index_value;

    int length() const { return (end - start); };

    static const index_range& find(const std::string& index);
    static std::vector<index_range> find(const std::vector<std::string>& indices);

    friend std::ostream& operator<<(std::ostream& o, index_range const& idx);
};

std::ostream& operator<<(std::ostream& o, index_range const& idx);

void declare_index_range(const std::string& name,
                         const std::string& indices,
                         const int& start,
                         const int& end);

std::vector<std::string> split_indices(const std::string& indices);

}}

#endif
