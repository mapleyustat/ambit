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

#include "iwl.h"
#include <util/exception.h>

namespace ambit { namespace io {

iwl::iwl(const std::string& full_pathname, enum OpenMode om, enum DeleteMode dm, double cutoff, bool psi34_compatible)
    : file(full_pathname, om, dm),
      nintegral(0),
      last_buffer(0),
      values(details::integrals_per_buffer__),
      p(details::integrals_per_buffer__),
      q(details::integrals_per_buffer__),
      r(details::integrals_per_buffer__),
      s(details::integrals_per_buffer__),
      labels_(details::integrals_per_buffer__ * 4),
      psi34_compatible_(psi34_compatible),
      cutoff_(cutoff),
      read_position_({0, 0})
{
    // ensure the iwl buffer exists in the file.
    if (om == kOpenModeOpenExisting) {
        if (toc().exists(details::buffer_key__) == false)
            throw std::runtime_error("IWL buffer does not exist in file: " + full_pathname);

        // go ahead and fetch the first buffer
        fetch();
    }
}

iwl::~iwl()
{}

void iwl::fetch()
{
    read_entry_stream(details::buffer_key__, read_position_, (int*)&last_buffer, 1);
    read_entry_stream(details::buffer_key__, read_position_, (int*)&nintegral, 1);

    if (psi34_compatible_)
        read_entry_stream(details::buffer_key__, read_position_, labels_.data(), 4 * details::integrals_per_buffer__);
    else
        throw ambit::util::not_implemented_error();

    read_entry_stream(details::buffer_key__, read_position_, values.data(), details::integrals_per_buffer__);

    // distribute the labels to their respective p, q, r, s
    for (int i = 0; i < details::integrals_per_buffer__; ++i) {
        p[i] = labels_[4*i+0];
        q[i] = labels_[4*i+1];
        r[i] = labels_[4*i+2];
        s[i] = labels_[4*i+3];
    }
}

}}
