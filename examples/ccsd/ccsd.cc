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
#include <iomanip>

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

    ambit::tensor::declare_index_range("occupied", "i,j,k,l,m,n", {0}, {closed});
    ambit::tensor::declare_index_range("virtual", "a,b,c,d,e,f", {closed}, {nmo});
    ambit::tensor::declare_index_range("so", "p,q,r,s", 0, nso);
    ambit::tensor::declare_index_range("mo", "u,v,x,y", 0, nmo);

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

        ambit::tensor::tensor C("MO Coefficients", "p,u");
        ambit::io::iwl::read_one(file32, "::MO coefficients", C, false);
//        std::cout << "C coefficients\n";
//        C.print();

//        ambit::tensor::tensor Co = C.slice("p,i");
//        std::cout << "C occupied coefficients\n";
//        Co.print();

//        ambit::tensor::tensor Cv = C.slice("u,a");
//        std::cout << "C virtual coefficients\n";
//        Cv.print();

        double Escf;
        file32.read("::SCF energy", &Escf, 1);
//        std::cout.width(19);
        std::cout << std::setprecision(14) << std::fixed;
        std::cout << "SCF Reference Energy   :   "   << Escf << "\n";

        // Read the SO basis Fock matrix
        ambit::tensor::tensor F_SO("SO basis Fock matrix", "p,q");
        ambit::io::iwl::read_one(file32, "::Fock Matrix", F_SO);
//        std::cout << "SO basis Fock matrix \n";
//        F_SO.print();



        //Transform Fock matrix into MO basis
        ambit::tensor::tensor F("MO basis Fock matrix", "u,v");
        {
            ambit::tensor::tensor Ftmp("Ftmp", "p,v");
            Ftmp["pv"] = C["qv"] * F_SO["pq"];
            F["uv"]    = C["pu"] * Ftmp["pv"];
        }
//        std::cout << "MO basis Fock matrix \n";
//        F.print();


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
//        Dijab.print();

        ambit::tensor::tensor Dia("Dia", "i,a");
        {
            ambit::tensor::tensor Dtmp("Dia","i,a");
            Dtmp["ia"] += Dii["i"];
            Dtmp["ia"] -= Daa["a"];
            Dia["ia"]  = 1.0 / Dtmp["ia"];
        }
//        Dia.print();
        // Modify tensor to take a lambda function that knows how to construct
        // the data on the fly.
        ambit::tensor::tensor Gmo("MO basis 2e integrals (uv|xy)", "u,v,x,y");
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
            ambit::tensor::tensor Gpqry("g_pqry", "p,q,r,y");
            ambit::tensor::tensor Gpqxy("g_pqxy", "p,q,x,y");
            ambit::tensor::tensor Gpvxy("g_pvxy", "p,v,x,y");

            Gpqry["pqry"] = C["sy"] * G["pqrs"];
            Gpqxy["pqxy"] = C["rx"] * Gpqry["pqry"];
            Gpvxy["pvxy"] = C["qv"] * Gpqxy["pqxy"];
            Gmo["uvxy"]   = C["pu"] * Gpvxy["pvxy"];
        }


        // Sort the integrals into physist's notation (uv|xy) -> <ux|vy>
        ambit::tensor::tensor G_p("g_uxvy", "u,x,v,y");
        G_p["uxvy"] = Gmo["uvxy"];

//        // Antisymmetrize
//        ambit::tensor::tensor Aijab("a_ijab", "i,j,a,b");
//        Aijab["ijab"] = Gijab["ijab"] - Gijab["ijba"];

//        // Squaring the terms
//        ambit::tensor::tensor Gijab2("ijab2", "i,j,a,b");
//        ambit::tensor::tensor Aijab2("a_ijab2", "i,j,a,b");
//        Aijab2["ijab"] = Aijab["ijab"] * Aijab["ijab"];
//        Gijab2["ijab"] = Gijab["ijab"] * Gijab["ijab"];

//        // Dot product
//        double e_aa = Aijab2["ijab"].dot(Dijab["ijab"]) / 4.0;
//        double e_bb = e_aa;
//        double e_ab = Gijab2["ijab"].dot(Dijab["ijab"]);
//        double e_mp2 = e_aa + e_ab + e_bb;

        // Form initial T1 and T2 amplitudes
        ambit::tensor::tensor t1("t_ia","i,a");
        t1["ia"] = F["ia"] * Dia["ia"];
        ambit::tensor::tensor t2("t_ijab","i,j,a,b");
        t2["ijab"] = G_p["ijab"] * Dijab["ijab"];

        // Compute the MP2 energy for test
        double e_mp2;
        {
            ambit::tensor::tensor Ttmp("Ttmp","i,j,a,b");
            Ttmp["ijab"]= t2["ijab"] * 2 - t2["jiab"];
            e_mp2 = G_p["ijab"].dot(Ttmp["ijab"]);
        }
        std::cout << "MP2 Correlation Energy :    " << e_mp2 << "\n";

    }

    ambit::util::print::finalize();

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif

    return 0;
}
