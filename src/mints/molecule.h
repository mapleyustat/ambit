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

#if !defined(AMBIT_SRC_MINTS_MOLECULE)
#define AMBIT_SRC_MINTS_MOLECULE

#include <util/aligned.h>

namespace ambit {
namespace mints {

struct molecule
{
    molecule(int natom);

    void print() const;

    void set_atom_x(const std::vector<double>& x);
    void set_atom_y(const std::vector<double>& y);
    void set_atom_z(const std::vector<double>& z);

    double nuclear_repulsion_energy() const;

private:
    /// The total number of atoms
    int natom_;
    /// Atomic charge of the atoms. Aligned memory.
    aligned_vector<double> atom_Z_;
    /// The x-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_x_;
    /// The y-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_y_;
    /// The z-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_z_;
};

}
}

#endif

