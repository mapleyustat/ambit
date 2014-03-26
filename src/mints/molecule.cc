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

#include "molecule.h"

#include <cmath>

namespace ambit {
namespace mints {

molecule::molecule(int natom)
    : natom_(natom), atom_Z_(natom, 0.0), atom_x_(natom, 0.0), atom_y_(natom, 0.0), atom_z_(natom, 0.0)
{
}

void molecule::print() const
{
    printf("molecule object\n");
    printf("=================\n");
    printf("natom_ = %d\n", natom_);
    printf("&atom_Z_ = %X\n", atom_Z_.data());
    printf("&atom_x_ = %X\n", atom_x_.data());
    printf("&atom_y_ = %X\n", atom_y_.data());
    printf("&atom_z_ = %X\n", atom_z_.data());
}

double molecule::nuclear_repulsion_energy() const
{
    double e = 0.0;

    double *x = (double*)__builtin_assume_aligned(atom_x_.data(), 16);
    double *y = (double*)__builtin_assume_aligned(atom_y_.data(), 16);
    double *z = (double*)__builtin_assume_aligned(atom_z_.data(), 16);

    for (int i=1; i<natom_; ++i) {
        ALIGNED_LOOP(int j=0; j<i; ++j) {
            double xdist = pow(x[i] - x[j], 2.0);
            double ydist = pow(y[i] - y[j], 2.0);
            double zdist = pow(z[i] - z[j], 2.0);
            double sum = xdist + ydist + zdist;
            double distance = sqrt(sum);

            e += atom_Z_[i] * atom_Z_[j] / distance;
        }
    }

    return e;
}

}
}
