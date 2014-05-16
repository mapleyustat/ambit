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

#include <tensor/indices.h>
//#include <tensor/tensor.h>
#include <tensor/cyclops/tensor.h>

//#include <tensor/dense_tensor.h>
//#include <tensor/composite_tensor.h>

#include <util/prettyprint.h>
#include <util/string.h>
#include <util/print.h>

#include <io/io.h>

#include <iostream>

int main(int argc, char** argv)
{
#if defined(HAVE_MPI)
    MPI::Init(argc, argv);
#endif // defined(HAVE_MPI)

    ambit::util::print::initialize();

    ambit::io::manager manager(".");
    ambit::io::file file32 = manager.scratch_file("psi.32");

    int nirrep = 0;
    file32.read("::Num. irreps", &nirrep, 1);
    std::cout << "nirrep = " << nirrep << "\n";

    int nso = 0;
    file32.read("::Num. SO", &nso, 1);
    std::cout << "nso = " << nso << "\n";

    ambit::tensor::declare_index_range("occupied", "i,j,k,l", {0}, {5});
    ambit::tensor::declare_index_range("virtual", "a,b,c,d", {3}, {4});
    ambit::tensor::declare_index_range("so", "p,q,r,s", 0, nso);

    //for (auto iter = ambit::tensor::index_range::set.begin(); iter != ambit::tensor::index_range::set.end(); ++iter) {
    //    std::cout << "name " << iter->first
    //              << " index " << iter->second.name
    //              << " start " << iter->second.start
    //              << " end " << iter->second.end
    //              << " value " << static_cast<int>(iter->second.index_value)
    //              << std::endl;
    //}

    //std::cout << "found " << ambit::tensor::index_range::find("i").start << std::endl;

    //std::vector<ambit::tensor::index_range> range = ambit::tensor::index_range::find(ambit::tensor::split_indices("i,j,a"));
    //for (auto& i : range) {
    //    std::cout << " index " << i.name
    //              << " start " << i.start
    //              << " end " << i.end
    //              << " value " << static_cast<int>(i.index_value)
    //              << std::endl;
    //}

    // this was for testing tensor_old
    {
//    ambit::tensor::dense_tensor<double> A("A", "i,j");
//    ambit::tensor::dense_tensor<double> B("B", "i,a");
//    ambit::tensor::dense_tensor<double> C("C", "j,a");

//    A.fill_with_random_data();
//    B.fill_with_random_data();

//    A.print();
//    B.print();

//    C["ja"] = A["ij"] * B["ia"];

//    C.print();

//    ambit::tensor::DenseTensor<double> B("B", {5, 5});
//    ambit::tensor::DenseTensor<double> C("C", {5, 5});

//    C["ij"] = A["ik"] * B["jk"];

//    C.print();
    }

    // start testing the new tensor library
    {
        ambit::tensor::tensor it0("test", "i,j");
        ambit::tensor::tensor it1("test", "i,j");
        ambit::tensor::tensor it2("test", "i,j");

        const ambit::tensor::index_range& occupied = ambit::tensor::index_range::find("i");

        //std::string indices = it.implicit();
        //std::cout << "indices " << indices << "\n";

        it1.fill_random();
        it2.fill_random();

        it0.multiply(1.0, it1["ik"], it2["kj"], 0.0, "ij");

        it0.print();

        it0["ij"] = it1["ik"] * it2["kj"];
        it0.print();

        ambit::tensor::key_generator2 ij(occupied, occupied);
        std::cout << "(0, 1) " << ij(0,1) << "\n";
        std::cout << "(1, 0) " << ij(1,0) << "\n";

        std::vector<tkv_pair<double>> values(10);
        int64_t count=0;
        for (auto& i : values) {
            i.k = ij(0, count);
            i.d = (double)count++;
        }
        it0.write(values);
        it0.print();

        // This does not work.
        //it0["ij"] = it1["ij"];
        //it0.print();
        //it0 = it1;
        //it0.print();
    }

    {
        ambit::io::file file35 = manager.scratch_file("psi.35");
        file35.toc().print();

        ambit::tensor::tensor S("overlap", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Overlap Ints", S);
        std::cout << "overlap\n";
        S.print();

        ambit::tensor::tensor T("kinetic", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Kinetic Energy Ints", T);
        std::cout << "kinetic\n";
        T.print();

        ambit::tensor::tensor V("potential", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Potential Energy Ints", V);
        std::cout << "potential\n";
        V.print();
    }

    ambit::util::print::finalize();

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif

    return 0;
}
