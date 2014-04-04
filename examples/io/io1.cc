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

#include <io/io.h>
#include <util/print.h>

int main(int argc, char **argv)
{
    // this call is MPI-aware and safe
    if (!ambit::util::print::initialize())
        printf("Unable to initialize print system.\n");

    ambit::util::print0("size of an entry structure is %lu\n", sizeof(ambit::io::toc::entry));
    // this tests ready PSI4 created files.
//    {
//        ambit::io::file file32("psi.32", ambit::io::kOpenModeOpenExisting);

//        // read in number of atoms
//        int natom = 0;
//        ambit::io::toc::entry& entry = file32.toc().entry("::Num. atoms");
//        file32.read(entry, &natom, 1);
//        ambit::util::print0("number of atoms %d\n", natom);

//        // read in number of atoms
//        natom = 0;
//        file32.read("::Num. atoms", &natom, 1);
//        ambit::util::print0("number of atoms %d\n", natom);

//        // read in something that doesn't exist
//        try {
//            file32.read("::Number of atoms", &natom, 1);
//            ambit::util::print0("should not have reached here\n");
//        }
//        catch (std::exception& e) {
//            ambit::util::print0("exception occurred: %s\n", e.what());
//        }

//        // attempt to read past the entry end point
//        try {
//            int atoms[2];
//            file32.read("::Num. atoms", &atoms, 2);
//            ambit::util::print0("should not have reached here\n");
//        }
//        catch (std::exception& e) {
//            ambit::util::print0("exception occurred: %s\n", e.what());
//        }
//    }

//    ambit::io::file test32("test.32", ambit::io::kOpenModeOpenExisting);
//    test32.toc().print();
//    test32.toc().write();

//    ambit::util::print0("size of Label: %lu\n", test32.toc().size("::Label"));

    // this tests creating new files, writing, reading values from it.
//    {
//        ambit::io::file new32("new.32", ambit::io::kOpenModeCreateNew);
//        ambit::io::file new32("new.32", ambit::io::kOpenModeOpenExisting);

//        ambit::io::toc::entry& entry = new32.toc().entry("New Entry");
//        double one = 1.0;
//        new32.write(entry, &one, 1);
//        double value = 0.0;
//        new32.read(entry, &value, 1);
//        ambit::util::print0("value is %lf\n", value);
//        new32.toc().print();
//    }

    // this tests the io system using the manager.
    {
        char label[81];
        ambit::io::manager manager(".");
        ambit::io::file file32 = manager.file("psi.32");
        file32.toc().print();

        file32.read("::Label", label, 80);
        label[80] = '\0';
        ambit::util::print0("Label is '%s'\n", label);
    }
    ambit::util::print::finalize();
}