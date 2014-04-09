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

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

#include <mints/molecule.h>
#include <mints/basisset.h>
#include <util/property_tree.h>
#include <util/print.h>
#include <util/prettyprint.h>
#include <util/string.h>

#include <iostream>

int main(int argc, char** argv)
{
#if defined(HAVE_MPI)
    MPI::Init(argc, argv);
#endif // defined(MPI)

    if (!ambit::util::print::initialize())
        printf("Unable to initialize print system.\n");

    ambit::mints::molecule molecule("water.xyz");
    molecule.print();

    ambit::mints::basisset basis("sto-3g", molecule);
    basis.print();

    ambit::util::print::finalize();

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif // defined MPI

    return 0;
}
