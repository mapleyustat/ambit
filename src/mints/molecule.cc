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
#include <util/print.h>

#include <cmath>

#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>
#include <exception>

namespace ambit {
namespace mints {

molecule::molecule(const std::string& xyzfile)
    : natom_(0)
{
    load_xyz_file(xyzfile);
}

void molecule::print() const
{
    util::print0("molecule object\n");
    util::print0("=================\n");
    util::print0("natom_ = %d\n", natom_);
//    printf("&atom_Z_ = %X\n", atom_Z_.data());
//    printf("&atom_x_ = %X\n", atom_x_.data());
//    printf("&atom_y_ = %X\n", atom_y_.data());
//    printf("&atom_z_ = %X\n", atom_z_.data());
}

double molecule::nuclear_repulsion_energy() const
{
    double e = 0.0;

    //double *x = (double*)__builtin_assume_aligned(atom_x_.data(), 16);
    //double *y = (double*)__builtin_assume_aligned(atom_y_.data(), 16);
    //double *z = (double*)__builtin_assume_aligned(atom_z_.data(), 16);

    const double *x = atom_x_.data();
    const double *y = atom_y_.data();
    const double *z = atom_z_.data();

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

bool molecule::load_xyz_file(const std::string& xyzfile, bool throw_on_error)
{
    if (xyzfile.empty()) {
        if (throw_on_error)
            throw std::runtime_error("Molecule::load_xyz_file: empty file name.");
        else
            return false;
    }

    std::ifstream infile(xyzfile.c_str());
    std::string line;
    const std::string bohr("bohr"), au("au");
    bool angstrom_in_file = true;

    if (!infile) {
        if (throw_on_error)
            throw std::runtime_error("Molecule::load_xyz_file: Unable to open " + xyzfile);
        else
            return false;
    }

    // read in first line.
    std::getline(infile, line);

    // this is what we expect to find on the first line
    boost::regex rx("(\\d+)\\s*(bohr|au)?", boost::regbase::normal | boost::regbase::icase);
    boost::smatch what;

    // try to match the first line
    if (boost::regex_match(line, what, rx)) {
        // matched
        natom_ = boost::lexical_cast<int>(what[1]);

        if (what.size() == 3) {
            std::string s(what[2].first, what[2].second);
            if (boost::iequals(bohr, s) || boost::iequals(au, s))
                angstrom_in_file = false;
        }
    }
    else {
        if (throw_on_error)
            throw std::runtime_error("Molecule::load_xyz_file: Malformed first line.");
        else
            return false;
    }

    // since we know the number of atoms, let's resize our data arrays
    atom_Z_.resize(natom_);
    atom_x_.resize(natom_);
    atom_y_.resize(natom_);
    atom_z_.resize(natom_);

    // the next line is a comment line, ignore it
    std::getline(infile, line);

    // now begins the useful information
    // here is the regex for matching the remaining lines
    rx.assign("(?:\\s*)([A-Z](?:[a-z])?)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s*)", boost::regbase::normal | boost::regbase::icase);

    for (int i=0; i<natom_; ++i) {
        // get an atom info line
        std::getline(infile, line);

        // attempt to match it
        if (boost::regex_match(line, what, rx)) {
            // first is a string
            std::string atom_sym(what[1].first, what[1].second);
            atom_Z_[i] = 1.0;

            // then the coordinates
            atom_x_[i] = boost::lexical_cast<double>(what[2]);
            atom_y_[i] = boost::lexical_cast<double>(what[3]);
            atom_z_[i] = boost::lexical_cast<double>(what[4]);
        }
    }
    return true;
}

}
}
