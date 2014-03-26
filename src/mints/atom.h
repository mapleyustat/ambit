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

#if !defined(AMBIT_LIB_MINTS_ATOM)
#define AMBIT_LIB_MINTS_ATOM

#include <array>
#include <string>

#include <boost/serialization/access.hpp>

namespace ambit {
namespace mints {

struct atom
{
protected:
    std::string name_;
    std::array<double, 3> position_;
    double charge_;
    double Z_;

private:
    friend class boost::serialization::access;
    template<typename Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar & name_ & position_;
    }
public:
    atom() {}

    atom(const std::string& name, const std::array<double, 3>& position, const double& charge);

    const std::string& name() const { return name_; }
    double charge() const { return charge_; }

    const std::array<double, 3>& position() const { return position_; }
    double position(const unsigned int i) const { return position_[i]; }

    bool operator==(const atom&) const;
};

}} // namespace ambit::mints

#endif

