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

#if !defined(AMBIT_SRC_IO_IWL)
#define AMBIT_SRC_IO_IWL

#include "file.h"
#include <util/aligned.h>
#include <util/constants.h>

namespace ambit { namespace io {

namespace details {

/// The number of integrals per batch to be read in.
static constexpr int integrals_per_buffer__ = 2980;
/// The label in the integral file to use. Should probably be abstracted away but this is what PSI3/4 uses.
static constexpr const char* buffer_key__ = "IWL Buffers";

}

struct iwl : public file
{
    iwl(const std::string& full_pathname, enum OpenMode om, enum DeleteMode dm = kDeleteModeKeepOnClose, double cutoff = numerical_zero__, bool psi34_compatible = true);
    virtual ~iwl();

    /// number of integrals valid in values, p, q, r, and s.
    const int nintegral;

    /// is this the last buffer?
    const int last_buffer;

    // implements SOA.
    aligned_vector<double> values;
    aligned_vector<short int> p;
    aligned_vector<short int> q;
    aligned_vector<short int> r;
    aligned_vector<short int> s;

    /// fetches the next batch of integrals.
    void fetch();

private:

    /// psi3/4 compatible label structure.
    aligned_vector<short int> labels_;

    const bool psi34_compatible_;
    const double cutoff_;

    address read_position_;
};

}}

#endif
