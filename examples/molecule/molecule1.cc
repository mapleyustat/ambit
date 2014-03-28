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

#include <mints/molecule.h>
#include <mints/basisset.h>
#include <util/property_tree.h>

#include <util/prettyprint.h>
#include <util/string.h>

#include <iostream>

int main(int /*argc*/, char** /*argv*/)
{
    ambit::mints::molecule molecule("water.xyz");
    molecule.print();

    ambit::mints::basisset basis("sto-3g", molecule);
    return 0;
}
