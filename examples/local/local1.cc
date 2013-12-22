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

#include <tensor/dense_tensor.h>
#include <tensor/composite_tensor.h>

int main(int argc, char** argv)
{
    ambit::tensor::DenseTensor<double> A("A", {5, 5});
    ambit::tensor::DenseTensor<double> B("B", {5, 5});
    ambit::tensor::DenseTensor<double> C("C", {5, 5});

    C["ij"] = A["ik"] * B["jk"];

    C.print();

    return 0;
}
