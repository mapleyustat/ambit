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

#include "atom.h"

#include <util/constants.h>

#include <cmath>

namespace ambit {
namespace mints {

atom::atom(const std::string& name, const std::array<double, 3>& position, const double& charge)
    : name_(name), position_(position), charge_(charge)
{
}

bool atom::operator==(const atom& other) const
{
    bool out = true;

    out &= name_ == other.name_;
    out &= fabs(position_[0]-other.position_[0]) < numerical_zero__;
    out &= fabs(position_[1]-other.position_[1]) < numerical_zero__;
    out &= fabs(position_[2]-other.position_[2]) < numerical_zero__;
    out &= charge_ == other.charge_;

    return out;
}

}
}
