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

#if !defined(MINTS_LIB_UTIL_PROPERTY_TREE)
#define MINTS_LIB_UTIL_PROPERTY_TREE

#include <array>
#include <sstream>
#include <memory>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast.hpp>

#include "aligned.h"

namespace ambit {
namespace util {

// Forward declaration
struct property_tree;

namespace detail {

struct property_tree_iterator
{
private:
    boost::property_tree::ptree::const_iterator current;

public:
    property_tree_iterator() {}
    explicit property_tree_iterator(boost::property_tree::ptree::const_iterator curr)
        : current(curr)
    {}

    // dereference operator - return the current node's data
    const property_tree operator*();

    // prefix returns by reference
    property_tree_iterator& operator++() { ++current; return *this; }
    property_tree_iterator& operator--() { --current; return *this; }

    // postfix should be implemented in terms of prefix operators
    property_tree_iterator& operator++(int) { property_tree_iterator& out = *this; ++*this; return out; }
    property_tree_iterator& operator--(int) { property_tree_iterator& out = *this; --*this; return out; }

    bool operator==(const property_tree_iterator& o) const { return current == o.current; }
    bool operator!=(const property_tree_iterator& o) const { return current != o.current; }
};

struct property_tree_reverse_iterator
{
private:
    boost::property_tree::ptree::const_reverse_iterator current;

public:
    property_tree_reverse_iterator() {}
    explicit property_tree_reverse_iterator(boost::property_tree::ptree::const_reverse_iterator curr)
        : current(curr)
    {}

    // dereference operator - return the current node's data
    const property_tree operator*();

    // prefix returns by reference
    property_tree_reverse_iterator& operator++() { ++current; return *this; }
    property_tree_reverse_iterator& operator--() { --current; return *this; }

    // postfix should be implemented in terms of prefix operators
    property_tree_reverse_iterator& operator++(int) { property_tree_reverse_iterator& out = *this; ++*this; return out; }
    property_tree_reverse_iterator& operator--(int) { property_tree_reverse_iterator& out = *this; --*this; return out; }

    bool operator==(const property_tree_reverse_iterator& o) const { return current == o.current; }
    bool operator!=(const property_tree_reverse_iterator& o) const { return current != o.current; }
};

} // namespace detail

struct property_tree
{
    typedef detail::property_tree_iterator iterator;
    typedef detail::property_tree_reverse_iterator reverse_iterator;

protected:
    boost::property_tree::ptree data_;
    std::string key_;

    // serialization
    friend class boost::serialization::access;
    template<class archive>
    void serialize(archive& ar, const unsigned int) {
        ar & data_ & key_;
    }

public:
    property_tree() : data_() {}

    property_tree(const boost::property_tree::ptree& i, const std::string& key)
        : data_(i), key_(key) {}

    property_tree(const property_tree& o)
        : data_(o.data_), key_(o.key_) {}

    property_tree(const std::string& input);

    property_tree get_child(const std::string& key) const {
        return property_tree(data_.get_child(key), key);
    }

    std::shared_ptr<property_tree> get_child_optional(const std::string& key) const {
        auto out = data_.get_child_optional(key);
        return out ? std::make_shared<property_tree>(*out, key) : nullptr;
    }

    template<typename T>
    T get(const std::string& s) const {
        return data_.get<T>(s);
    }

    template<typename T>
    T get(const std::string& s, const T& t) const {
        return data_.get<T>(s, t);
    }

    void add_child(const std::string& s, std::shared_ptr<property_tree> ch) {
        data_.add_child(s, ch->data_);
    }

    template<typename T>
    void put(const std::string& s, const T& o) {
        data_.put<T>(s, o);
    }

    template<typename T>
    void push_back(const T& o) {
        assert(typeid(T) != typeid(std::shared_ptr<property_tree>));
        boost::property_tree::ptree ch;
        ch.put("", boost::lexical_cast<std::string>(o));
        data_.push_back(std::make_pair("", ch));
    }

    template<typename T>
    aligned_vector<T> get_aligned_vector(const std::string& key) const;
    template<typename T>
    std::vector<T> get_vector(const std::string& s, const int nexpected = 0) const;
    template<typename T, int N>
    std::array<T,N> get_array(const std::string& s) const;

    void erase(const std::string& key) { data_.erase(key); }

    std::string data() const { return data_.data(); }
    std::string key() const { return key_; }

    size_t size() const;

    iterator begin() const;
    iterator end() const;

    reverse_iterator rbegin() const;
    reverse_iterator rend() const;

    void print() const;

    // static function to read basis files
    static property_tree read_basis(std::string name);
};

template<>
void property_tree::push_back<std::shared_ptr<property_tree>>(const std::shared_ptr<property_tree>& pt);

template<typename T>
aligned_vector<T> property_tree::get_aligned_vector(const std::string& key) const
{
    auto tmp = get_child(key);
    aligned_vector<T> out(tmp.size());
    int count = 0;
    for (auto& i : tmp)
        out[count++] = boost::lexical_cast<T>(i.data());
    return out;
}

template<typename T>
std::vector<T> property_tree::get_vector(const std::string& key, const int nexpected) const
{
    std::vector<T> out;
    auto tmp = get_child(key);
    if ( (nexpected > 0) && (tmp.size() != nexpected) ) {
        std::stringstream err;
        err << "Unexpected number of elements in vector " << key << ". Expected: " << nexpected << ", received: " << tmp.size();
        throw std::runtime_error(err.str());
    }
    for (auto& i : tmp)
        out.push_back(boost::lexical_cast<T>(i.data()));
    return out;
}

template<typename T, int N>
std::array<T,N> property_tree::get_array(const std::string& key) const
{
    std::array<T,N> out;
    auto tmp = get_child(key);
    if (tmp.size() != N) {
        std::stringstream err;
        err << "Unexpected number of elements in array " << key << ". Expected: " << N << ", received: " << tmp.size();
        throw std::runtime_error(err.str());
    }
    int n = 0;
    for (auto& i : tmp)
        out[n++] = boost::lexical_cast<T>(i.data());
    return out;
}

}} // namespace ambit::util

#endif

