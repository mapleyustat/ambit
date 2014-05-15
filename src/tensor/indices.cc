/*
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

#include "indices.h"
#include "exception.h"

#include <util/string.h>
#include <util/prettyprint.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

namespace ambit { namespace tensor {

index_range::set_type index_range::set;

const index_range& index_range::find(const std::string& idx)
{
    auto it = set.find(idx);
    if (it == set.end())
        throw index_not_found_error();
    return it->second;
}

std::vector<index_range> index_range::find(const std::vector<std::string>& indices)
{
    std::vector<index_range> v;
    for (auto& i : indices)
        v.push_back(find(i));
    return v;
}

void declare_index_range(const std::string& name_,
                         const std::string& indices,
                         const int& start,
                         const int& end)
{
    std::istringstream f(indices);
    std::string s;
    std::string name = name_;
    util::trim(name);
    std::vector<std::string> v = split_indices(indices);

    for (auto i : v) {
        // before adding make sure index does not already exist in set
        if (index_range::set.find(i) != index_range::set.end())
            throw index_already_exists_error();

        index_range r;
        r.name = name;
        r.start = start;
        r.end = end;

        // unique value for this index range
        r.index_value = static_cast<char>(index_range::set.size());

        // do not call set[i] = r. that performs a constructor call to first create set[i], then a reference is pass back and operator= is called for r.
        index_range::set.insert(std::make_pair(i, r));
    }
}

std::vector<std::string> split_indices(const std::string& indices)
{
    std::istringstream f(indices);
    std::string s;
    std::vector<std::string> v;

    if (indices.find(",") != std::string::npos) {
        while (std::getline(f, s, ',')) {
            std::string trimmed = util::trim(s);
            v.push_back(trimmed);
        }
    }
    else {
        // simply split the string up
        for (int i=0; i<indices.size(); ++i)
            v.push_back(std::string(1, indices[i]));
    }

    return v;
}

std::ostream& operator<<(std::ostream& o, index_range const& idx)
{
    o << idx.name << " start: " << idx.start << " end: " << idx.end;
    return o;
}

}}
