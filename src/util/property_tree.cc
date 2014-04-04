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

#include "property_tree.h"

#include <fstream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace ambit { namespace util {

namespace detail {

const property_tree property_tree_iterator::operator*()
{
    return property_tree(current->second, current->first);
}

const property_tree property_tree_reverse_iterator::operator*()
{
    return property_tree(current->second, current->first);
}

} // namespace detail

property_tree::property_tree(const std::string& input)
{
    // Attempt to use file extension
    const size_t n = input.find_last_of(".");
    const std::string extension = (n != std::string::npos) ? input.substr(n) : "";

    if (extension == ".json")
        boost::property_tree::json_parser::read_json(input, data_);
    else if (extension == ".xml")
        boost::property_tree::xml_parser::read_xml(input, data_);
    else { // unable to determine type, try them all
        try {
            boost::property_tree::json_parser::read_json(input, data_);
        }
        catch (boost::property_tree::json_parser_error& e) {
            try {
                boost::property_tree::xml_parser::read_xml(input, data_);
            }
            catch (boost::property_tree::xml_parser_error& f) {
                throw std::runtime_error("Failed to determine input file format.");
            }
        }
    }
}

property_tree::iterator property_tree::begin() const
{
    return iterator(data_.begin());
}

property_tree::iterator property_tree::end() const
{
    return iterator(data_.end());
}

property_tree::reverse_iterator property_tree::rbegin() const
{
    return reverse_iterator(data_.rbegin());
}

property_tree::reverse_iterator property_tree::rend() const
{
    return reverse_iterator(data_.rend());
}

template<>
void property_tree::push_back<std::shared_ptr<property_tree>>(const std::shared_ptr<property_tree>& pt)
{
    data_.push_back(std::make_pair("", pt->data_));
}

size_t property_tree::size() const
{
    return data_.size();
}

void property_tree::print() const
{
    write_json(std::cout, data_);
}

property_tree property_tree::read_basis(std::string name)
{
    boost::to_lower(name);

    try {
        return property_tree(std::string(ROOT_SRC_DIR) + "/basis/" + name + ".json");
    }
    catch (...) {
        throw std::runtime_error(name + " cannot be opened. Please specify the full path to the basis file.");
    }
}

}} // namespace ambit::util

