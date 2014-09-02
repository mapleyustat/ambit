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
    file32.toc().print();

    int nirrep = 0;
    file32.read("::Num. irreps", &nirrep, 1);
    std::cout << "nirrep = " << nirrep << "\n";

    int nso = 0;
    file32.read("::Num. SO", &nso, 1);
    std::cout << "nso = " << nso << "\n";

    int nmo = 0;
    file32.read("::Num. MO's", &nmo, 1);
    std::cout << "nmo = " << nmo << "\n";

    int closed = 0;
    file32.read("::Closed shells per irrep", &closed, 1);
    std::cout << "clsd = " << closed << "\n";

    ambit::tensor::declare_index_range("occupied", "i,j,k,l", {0}, {closed});
    ambit::tensor::declare_index_range("virtual", "a,b,c,d", {closed}, {nmo});
    ambit::tensor::declare_index_range("so", "p,q,r,s", 0, nso);
    ambit::tensor::declare_index_range("mo", "m,n", 0, nmo);

    std::vector<double> e(nmo);
    file32.read("::MO energies", e);
    std::cout << "Orbital energies: ";
    for (int i=0; i<nmo; ++i)
        std::cout << e[i] << " ";
    std::cout << "\n";



    {
        ambit::io::file file35 = manager.scratch_file("psi.35");
//        file35.toc().print();

        ambit::tensor::tensor S("overlap", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Overlap Ints", S);
//        std::cout << "overlap\n";
//        S.print();

        ambit::tensor::tensor T("kinetic", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Kinetic Energy Ints", T);
//        std::cout << "kinetic\n";
//        T.print();

        ambit::tensor::tensor V("potential", "p,q");
        ambit::io::iwl::read_one(file35, "SO-basis Potential Energy Ints", V);
//        std::cout << "potential\n";
//        V.print();

        ambit::tensor::tensor C("MO Coefficients", "p,m");
        ambit::io::iwl::read_one(file32, "::MO coefficients", C, false);
//        std::cout << "C coefficients\n";
//        C.print();

        ambit::tensor::tensor Co = C.slice("p,i");
//        std::cout << "C occupied coefficients\n";
//        Co.print();

        ambit::tensor::tensor Cv = C.slice("p,a");
//        std::cout << "C virtual coefficients\n";
//        Cv.print();

        // Construct denominators
        ambit::tensor::tensor Dii("Dii", "i");
        ambit::tensor::tensor Daa("Daa", "a");
        std::vector<tkv_pair<double>> temp(closed);
        for (int i=0; i<closed; i++) {
            temp[i].k = i;
            temp[i].d = e[i];
        }
        Dii.write(temp);
//        Dii.print();

        temp.resize(nmo-closed);
        for (int i=0; i < (nmo-closed); ++i) {
            temp[i].k = i;
            temp[i].d = e[i+closed];
        }
        Daa.write(temp);
//        Daa.print();

        ambit::tensor::tensor Dijab("Dijab", "i,j,a,b");
        {
            ambit::tensor::tensor Dtmp("Dijab", "i,j,a,b");
            Dtmp["ijab"] += Dii["i"];
            Dtmp["ijab"] += Dii["j"];
            Dtmp["ijab"] -= Daa["a"];
            Dtmp["ijab"] -= Daa["b"];
            Dijab["ijab"] = 1.0 / Dtmp["ijab"];
        }

        // Modify tensor to take a lambda function that knows how to construct
        // the data on the fly.
        ambit::tensor::tensor Giajb("g_iajb", "i,a,j,b");
        // For example this is how to transform the AO integrals to iajb.
        // In this case once we compute iajb we don't need to do it again,
        // but there are cases where we'll need to call this lambda each
        // time...in the case that C changes
        {
            // Load AO two-electron integrals from disk.
            // TODO: Make sure only the master node does this;
            //       need to modify read_two.
            ambit::tensor::tensor G("ERIs", "p,q,r,s");
            ambit::io::iwl iwl("psi.33", ambit::io::kOpenModeOpenExisting);
            ambit::io::iwl::read_two(iwl, G);
//            std::cout << "ERIs\n";
//            G.print();

            // Integral transformation
            ambit::tensor::tensor Gpqrb("g_pqrb", "p,q,r,b");
            ambit::tensor::tensor Gpqjb("g_pqjb", "p,q,j,b");
            ambit::tensor::tensor Gpajb("g_pajb", "p,a,j,b");

            Gpqrb["pqrb"] = Cv["sb"] * G["pqrs"];
            Gpqjb["pqjb"] = Co["rj"] * Gpqrb["pqrb"];
            Gpajb["pajb"] = Cv["qa"] * Gpqjb["pqjb"];
            Giajb["iajb"] = Co["pi"] * Gpajb["pajb"];
        }
//        std::cout << "(ia|jb)\n";
//        Giajb.print();

        // Sort the integrals
        ambit::tensor::tensor Gijab("g_ijab", "i,j,a,b");
        Gijab["ijab"] = Giajb["iajb"];

        // Antisymmetrize
        ambit::tensor::tensor Aijab("a_ijab", "i,j,a,b");
        Aijab["ijab"] = Gijab["ijab"] - Gijab["ijba"];

        // Squaring the terms
        ambit::tensor::tensor Gijab2("ijab2", "i,j,a,b");
        ambit::tensor::tensor Aijab2("a_ijab2", "i,j,a,b");
        Aijab2["ijab"] = Aijab["ijab"] * Aijab["ijab"];
        Gijab2["ijab"] = Gijab["ijab"] * Gijab["ijab"];

        // Dot product
        double e_aa = Aijab2["ijab"].dot(Dijab["ijab"]) / 4.0;
        double e_bb = e_aa;
        double e_ab = Gijab2["ijab"].dot(Dijab["ijab"]);
        double e_mp2 = e_aa + e_ab + e_bb;
        std::cout.width(19);
        std::cout.precision(14);
        std::cout << "MP2 Correlation Energy:    " << e_mp2 << "\n";
    }

    ambit::util::print::finalize();

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif

    return 0;
}
