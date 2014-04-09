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

#if !defined(AMBIT_SRC_MINTS_BASISSET)
#define AMBIT_SRC_MINTS_BASISSET

#include <util/aligned.h>

#include <string>

namespace ambit { namespace mints {

struct molecule;

struct basisset
{
    basisset(const std::string& name, const molecule& m);

    void print() const;

private:
    std::string name_;

    // Molecule this refers basis set refers to.
    const molecule& molecule_;

    /// The atom the shell is centered on.
    aligned_vector<int> atom_center_;
    /// The angular momentum component of the basis set
    aligned_vector<int> angular_momentum_;
    /// Exponents of the basis set blocked by shell.
    std::vector<aligned_vector<double>> exponents_;
    /// The coefficients of the basis set blocked by shell.
    std::vector<aligned_vector<double>> coefficients_;
};

}}

#endif

