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

#include <tensor/world.h>
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

namespace atom {

namespace {

const std::vector<std::string> symbols = {{
                                              "Gh",
                                              "H",
                                              "He",
                                              "Li",
                                              "Be",
                                              "B",
                                              "C",
                                              "N",
                                              "O",
                                              "F",
                                              "Ne"
                                          }};

}

std::string symbol_from_Z(int Z)
{
    assert(Z >= 0 && Z < symbols.size());
    return symbols[Z];
}

int Z_from_symbol(const std::string& symbol)
{
    assert(!symbol.empty());

    std::vector<std::string>::const_iterator found = std::find_if(symbols.begin(), symbols.end(), [=] (const std::string& s) { return boost::iequals(s, symbol); } );

    if (found == symbols.end())
        return -1;

    return std::distance(symbols.begin(), found);
}

}

molecule::molecule(const std::string& xyzfile)
    : natom_(0)
{
    load_xyz_file(xyzfile);
}

void molecule::print() const
{
    if (name_.length())
        util::printn("name = %s\n", name_.c_str());
    util::printn("repulsion = %20.15lf\n\n", nuclear_repulsion_energy());
    util::printn("   Center              X                  Y                   Z       \n");
    util::printn("------------   -----------------  -----------------  -----------------\n");
    for (int i=0; i<natom_; ++i) {
        util::printn("%8s%4s   %17.12f  %17.12f  %17.12f\n",atom_symbol_[i].c_str(), "", atom_x_[i], atom_y_[i], atom_z_[i]);
    }
}

double molecule::nuclear_repulsion_energy() const
{
    double e = 0.0;

    const double *x = atom_x_.data();
    const double *y = atom_y_.data();
    const double *z = atom_z_.data();

    for (int i=1; i<natom_; ++i) {
        for (int j=0; j<i; ++j) {
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
    // TODO: Abstract away tensor/cyclops/world.h from here.
    ambit::tensor::world& world = ambit::tensor::world::shared();

    if (world.rank == 0) {
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
        atom_symbol_.resize(natom_);
        atom_x_.resize(natom_);
        atom_y_.resize(natom_);
        atom_z_.resize(natom_);

        // the next line is a comment line, ignore it
        std::getline(infile, name_);

        // now begins the useful information
        // here is the regex for matching the remaining lines
        rx.assign("(?:\\s*)([A-Z](?:[a-z])?)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s+)(-?\\d+\\.\\d+)(?:\\s*)", boost::regbase::normal | boost::regbase::icase);

        for (int i=0; i<natom_; ++i) {
            // get an atom info line
            std::getline(infile, line);

            // attempt to match it
            if (boost::regex_match(line, what, rx)) {
                // first is a string of the atomic symbol
                std::string symbol(what[1].first, what[1].second);
                atom_symbol_[i] = symbol;
                atom_Z_[i] = atom::Z_from_symbol(symbol);

                // then the coordinates
                atom_x_[i] = boost::lexical_cast<double>(what[2]);
                atom_y_[i] = boost::lexical_cast<double>(what[3]);
                atom_z_[i] = boost::lexical_cast<double>(what[4]);
            }
        }
    }

    // broadcast results to all nodes.
    world.bcast(natom_, 0);
    world.bcast(name_, 0);
    world.bcast(atom_Z_, 0);
    world.bcast(atom_symbol_, 0);
    world.bcast(atom_x_, 0);
    world.bcast(atom_y_, 0);
    world.bcast(atom_z_, 0);

    return true;
}

}
}
