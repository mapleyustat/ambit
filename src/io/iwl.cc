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

void iwl::read_one(file& io, const std::string& label, ambit::tensor::tensor& tensor, bool symmetric)
{
    // ensure the tensor object is only 2D.
    if (tensor.dimension() != 2)
        throw std::runtime_error("tensor must be 2 dimension.");

    // psi stores lower triangle full block (may be symmetry blocked, but we don't care).
    const std::vector<ambit::tensor::index_range>& ir = tensor.index_ranges();

    ambit::tensor::key_generator2 key(ir[0], ir[1]);
    int n = ir[0].length();
    int ntri;
    if (symmetric)
        ntri = n * (n+1) / 2;
    else
        ntri = n*n;

    std::vector<double> ints(ntri);
    io.read(label, ints);

    std::vector<tkv_pair<double>> values(n*n);
    int64_t c=0, d=0;
    for (int p=0; p<ir[0].length(); ++p) {
        int p1;
        if (symmetric)
            p1 = p+1;
        else
            p1 = n;

        for (int q=0; q<p1; ++q) {
            values[c].k = key(q, p);
            values[c++].d = ints[d];

            //std::cout << "p " << p << " q " << q << " value " << ints[d] << "\n";

            if (symmetric && p != q) {
                values[c].k = key(p,q);
                values[c++].d = ints[d];
            }
            d++;
        }
    }
    tensor.write(values);
}

void iwl::read_two(iwl& io, ambit::tensor::tensor& tensor)
{
    // ensure the tensor object is only 4D.
    if (tensor.dimension() != 4)
        throw std::runtime_error("tensor must be 4 dimension.");

    // psi stores 1 of the 8 possible permutations
    const std::vector<ambit::tensor::index_range>& ir = tensor.index_ranges();

    ambit::tensor::key_generator4 key(ir[0], ir[1], ir[2], ir[3]);
    std::vector<tkv_pair<double>> values(8*details::integrals_per_buffer__);

    size_t count = 0;
    do {
        size_t c = 0;
        values.resize(8*details::integrals_per_buffer__);

        for (int i=0; i<io.nintegral; ++i) {

//            ambit::util::print0("read (%d %d| %d %d) = %lf\ncontributes to\n", io.p[i], io.q[i], io.r[i], io.s[i], io.values[i]);

            short int p, q, r, s;
            p = io.p[i]; q = io.q[i]; r = io.r[i]; s = io.s[i];

            // pppp case
            if (p == q && p == r && p == s) {
                values[c].k = key(p, p, p, p); values[c++].d = io.values[i];

//                ambit::util::print0("[%d %d %d %d]\n", p, p, p, p);
            }
            // pprr case
            else if (p == q && r == s) {
                values[c].k = key(p, p, r, r); values[c++].d = io.values[i];
                values[c].k = key(r, r, p, p); values[c++].d = io.values[i];

//                ambit::util::print0("[%d %d %d %d]\n", p, p, r, r);
//                ambit::util::print0("[%d %d %d %d]\n", r, r, p, p);
            }
            // pprs
            else if (p == q) {
                values[c].k = key(p, p, r, s); values[c++].d = io.values[i];
                values[c].k = key(p, p, s, r); values[c++].d = io.values[i];
                values[c].k = key(r, s, p, p); values[c++].d = io.values[i];
                values[c].k = key(s, r, p, p); values[c++].d = io.values[i];
//                ambit::util::print0("[%d %d %d %d]\n", p, p, r, s);
//                ambit::util::print0("[%d %d %d %d]\n", p, p, s, r);
//                ambit::util::print0("[%d %d %d %d]\n", r, s, p, p);
//                ambit::util::print0("[%d %d %d %d]\n", s, r, p, p);
            }
            // pqrr
            else if (r == s) {
                values[c].k = key(p, q, r, r); values[c++].d = io.values[i];
                values[c].k = key(q, p, r, r); values[c++].d = io.values[i];
                values[c].k = key(r, r, p, q); values[c++].d = io.values[i];
                values[c].k = key(r, r, q, p); values[c++].d = io.values[i];
//                ambit::util::print0("[%d %d %d %d]\n", p, q, r, r);
//                ambit::util::print0("[%d %d %d %d]\n", q, p, r, r);
//                ambit::util::print0("[%d %d %d %d]\n", r, r, p, q);
//                ambit::util::print0("[%d %d %d %d]\n", r, r, q, p);
            }
            else if (p == r && q == s) {
                values[c].k = key(p, q, p, q); values[c++].d = io.values[i];
                values[c].k = key(p, q, q, p); values[c++].d = io.values[i];
                values[c].k = key(q, p, p, q); values[c++].d = io.values[i];
                values[c].k = key(q, p, q, p); values[c++].d = io.values[i];
//                ambit::util::print0("[%d %d %d %d]\n", p, q, p, q);
//                ambit::util::print0("[%d %d %d %d]\n", p, q, q, p);
//                ambit::util::print0("[%d %d %d %d]\n", q, p, p, q);
//                ambit::util::print0("[%d %d %d %d]\n", q, p, q, p);
            }
            else {
                values[c].k = key(p, q, r, s); values[c++].d = io.values[i];
                values[c].k = key(p, q, s, r); values[c++].d = io.values[i];
                values[c].k = key(q, p, r, s); values[c++].d = io.values[i];
                values[c].k = key(q, p, s, r); values[c++].d = io.values[i];
                values[c].k = key(r, s, p, q); values[c++].d = io.values[i];
                values[c].k = key(r, s, q, p); values[c++].d = io.values[i];
                values[c].k = key(s, r, p, q); values[c++].d = io.values[i];
                values[c].k = key(s, r, q, p); values[c++].d = io.values[i];
//                ambit::util::print0("[%d %d %d %d]\n", p, q, r, s);
//                ambit::util::print0("[%d %d %d %d]\n", p, q, s, r);
//                ambit::util::print0("[%d %d %d %d]\n", q, p, r, s);
//                ambit::util::print0("[%d %d %d %d]\n", q, p, s, r);
//                ambit::util::print0("[%d %d %d %d]\n", r, s, p, q);
//                ambit::util::print0("[%d %d %d %d]\n", r, s, q, p);
//                ambit::util::print0("[%d %d %d %d]\n", s, r, p, q);
//                ambit::util::print0("[%d %d %d %d]\n", s, r, q, p);
            }
        }
        count += c;

        values.resize(c);

        tensor.write(values);

        if (io.last_buffer)
            break;
        io.fetch();
    } while(1);
}

}}
