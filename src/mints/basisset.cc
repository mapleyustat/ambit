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

#include <util/property_tree.h>
#include <tensor/cyclops/world.h>

#include <boost/property_tree/json_parser.hpp>

namespace ambit { namespace mints {

basisset::basisset(const std::string& name, const molecule& m)
    : name_(name), molecule_(m)
{
    // load in the basis set.
    // TODO: abstract cyclops world out
    ambit::tensor::cyclops::world& world = ambit::tensor::cyclops::world::shared();

    if (world.rank == 0) {
        ambit::util::property_tree basis_file(std::string(ROOT_SRC_DIR) + "/basis/" + name + ".json");

        for (int i = 0; i < m.natom(); ++i) {
            std::string a = m.symbol(i);
            std::transform(a.begin(), a.end(), a.begin(), ::tolower);
            a[0] = ::toupper(a[0]);

            ambit::util::property_tree atom_basis = basis_file.get_child(a);
            for (auto& ibasis : atom_basis) {
//                ibasis.print();
                exponents_.push_back(ibasis.get_aligned_vector<double>("prim"));
            }
        }
    }
}

}}
