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

#include "basisset.h"
#include "molecule.h"

#include <util/print.h>
#include <util/property_tree.h>
#include <tensor/world.h>

#include <boost/property_tree/json_parser.hpp>

namespace ambit { namespace mints {

namespace {

int angular_momentum_string_to_integer(const std::string& a)
{
    //                            a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v  w  x  y  z
    static char shell_to_am[] = {-1,-1,-1, 2,-1, 3, 4, 5, 6,-1, 7, 8, 9,10,11, 1,12,13, 0,14,15,16,17,18,19,20};
    int lower = tolower(a[0]) - 'a';

    if (lower < 0 || lower > 25)
        throw std::runtime_error("angular_momentum_string_to_integer: angular momentum string is out-of-bounds: " + a + "\n");

    if (shell_to_am[lower] == -1)
        throw std::runtime_error("angular_momentum_string_to_integer: unsupported angular momentum type: " + a + "\n");

    return shell_to_am[lower];
}

}

basisset::basisset(const std::string& name, const molecule& m)
    : name_(name), molecule_(m)
{
    // load in the basis set.
    ambit::tensor::world& world = ambit::tensor::world::shared();

    if (world.rank == 0) {
        ambit::util::property_tree basis_file(std::string(ROOT_SRC_DIR) + "/basis/" + name + ".json");

        for (int i = 0; i < m.natom(); ++i) {
            std::string a = m.symbol(i);
            std::transform(a.begin(), a.end(), a.begin(), ::tolower);
            a[0] = ::toupper(a[0]);

            // atom_basis will now contain the basis set information specifc for the atom symbol found in 'a'.
            ambit::util::property_tree atom_shells = basis_file.get_child(a);
            for (auto& atom_shell : atom_shells) {
                atom_shell.print();

                // in our basis set file the contraction coefficents can be group together if they share the same exponents
                ambit::util::property_tree ceofficients = atom_shell.get_child("cont");

                // for the shell grab the angular momentum
                const std::string ang = atom_shell.get<std::string>("angular");

                // record information needed for the shell
                atom_center_.push_back(i);
                angular_momentum_.push_back(angular_momentum_string_to_integer(ang));

                // loop over the groups of contraction coefficients
                exponents_.push_back(atom_shell.get_aligned_vector<double>("prim"));
                coefficients_.push_back(atom_shell.get_aligned_vector<double>("cont"));
            }
        }
    }
}

void basisset::print() const
{
    ambit::util::print::banner("AO Basis Set Information");
    {
        ambit::util::print::indenter indent;

        molecule_.print();

        for (int i=0; i<atom_center_.size(); ++i) {
            ambit::util::printn("%s: l %d\n", molecule_.symbol(atom_center_[i]).c_str(), angular_momentum_[i]);
            ambit::util::print::indenter indent1;

            for (int j=0; j<coefficients_[i].size(); ++j) {
                ambit::util::printn("%20.16lf %20.16lf\n", coefficients_[i][j], exponents_[i][j]);
            }
        }
    }
}

}}
