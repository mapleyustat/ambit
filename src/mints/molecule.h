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

#include <string>

namespace ambit {
namespace mints {

namespace atom {

std::string symbol_from_Z(int Z);

int Z_from_symbol(const std::string& symbol);

}

struct molecule
{
    molecule(const std::string& xyzfile);

    void print() const;

    int natom() const { return natom_; }
    std::string symbol(int atom) const { return atom_symbol_[atom]; }

    void set_atom_x(const std::vector<double>& x);
    void set_atom_y(const std::vector<double>& y);
    void set_atom_z(const std::vector<double>& z);

    double nuclear_repulsion_energy() const;

    bool load_xyz_file(const std::string& xyzfile, bool throw_on_error = true);

private:
    /// The total number of atoms
    int natom_;
    /// Actual atomic symbol (initially the one from the xyz file).
    std::vector<std::string> atom_symbol_;
    /// Atomic charge of the atoms. Aligned memory.
    aligned_vector<double> atom_Z_;
    /// The x-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_x_;
    /// The y-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_y_;
    /// The z-coordinates of the atoms. Aligned memory.
    aligned_vector<double> atom_z_;

    /** Name of the molecule.
     *
     * When a molecule is constructed with load_xyz_file then the name
     * comes from the comment line in the file.
     */
    std::string name_;
};

}
}

#endif

