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

#include <tensor/indices.h>
#include <tensor/dense_tensor.h>
#include <tensor/composite_tensor.h>

#include <util/prettyprint.h>
#include <util/string.h>

#include <iostream>

int main(int /*argc*/, char** /*argv*/)
{
    ambit::tensor::declare_index_range("occupied", "i,j,k,l", {0}, {5});
    ambit::tensor::declare_index_range("virtual", "a,b,c,d", {3}, {4});

    for (auto iter = ambit::tensor::index_range::set.begin(); iter != ambit::tensor::index_range::set.end(); ++iter) {
        std::cout << "name " << iter->first
                  << " index " << iter->second.name
                  << " start " << iter->second.start
                  << " end " << iter->second.end
                  << " value " << static_cast<int>(iter->second.index_value)
                  << std::endl;
    }

    std::cout << "found " << ambit::tensor::index_range::find("i").start << std::endl;

    std::vector<ambit::tensor::index_range> range = ambit::tensor::index_range::find(ambit::tensor::split_indices("i,j,a"));
    for (auto& i : range) {
        std::cout << " index " << i.name
                  << " start " << i.start
                  << " end " << i.end
                  << " value " << static_cast<int>(i.index_value)
                  << std::endl;
    }

    ambit::tensor::dense_tensor<double> A("A", "i,j");
    ambit::tensor::dense_tensor<double> B("B", "i,a");
    ambit::tensor::dense_tensor<double> C("C", "j,a");

    A.fill_with_random_data();
    B.fill_with_random_data();

    A.print();
    B.print();

    C["ja"] = A["ij"] * B["ia"];

    C.print();

//    ambit::tensor::DenseTensor<double> B("B", {5, 5});
//    ambit::tensor::DenseTensor<double> C("C", {5, 5});

//    C["ij"] = A["ik"] * B["jk"];

//    C.print();

    return 0;
}
