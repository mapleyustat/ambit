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
        ambit::tensor::tensor G_p("<ux|vy>", "u,x,v,y");
        G_p["uxvy"] = Gmo["uvxy"];

        /*
         * Slice the G tensor into 6 types of TEIs:
         * <ia|bc>, <ij|ab>, <ij|ka>, <ab|cd>, <ia|jb>, <ij|kl>
         * The last type <ij|kl> is not used in CCSD
         */

//        ambit::tensor::tensor Givxy = G_p.slice("i,a,b,c"); ?

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
            Ttmp["ijab"] = 2.0 * t2["ijab"];
            Ttmp["ijab"] -= t2["jiab"];
            e_mp2 = G_p["ijab"].dot(Ttmp["ijab"]);
        }
        std::cout << "MP2 Correlation Energy :    " << e_mp2 << "\n";

        /* Start CCSD Iteration */
        // RHF-CCSD Equations from Dr. Yamaguchi's Notes

        double Ecc = 0.0;
        int maxiter = 150;
        int E_conv = 8, t_conv = 8;

        for(int iter=0;iter<maxiter;++iter) {

            //Form the intermediates
            //Fae
            ambit::tensor::tensor Fae("Fae","a,b");
            Fae["ae"] = t1["mf"]*(2*G_p["amef"]-G_p["maef"]);
            Fae["ae"] -= (t2["mnaf"] + 0.5*t1["ma"]*t1["nf"])*(2*G_p["mnef"]-G_p["nmef"]);

            //Fmi
            ambit::tensor::tensor Fmi("Fmi","m,i");
            Fmi["mi"] = t1["mi"]*(2*G_p["mnie"]-G_p["nmie"]);
            Fmi["mi"] += (t2["inef"]+0.5*t1["ie"]*t1["nf"])*(2*G_p["mnef"]-G_p["nmef"]);

            //Fme
            ambit::tensor::tensor Fme("Fme","m,e");
            Fme["me"] = t1["nf"]*(2*G_p["mnef"]-G_p["nmef"]);

            //Wmnij
            ambit::tensor::tensor Wmnij("Wmnij","m,n,i,j");
            Wmnij["mnij"] = G_p["mnij"];
            Wmnij["mnij"] += t1["je"]*G_p["mnie"];
            Wmnij["mnij"] += t1["ie"]*G_p["mnej"];
            Wmnij["mnij"] += 0.5*(t2["ijef"]+t1["ie"]*t1["jf"])*G_p["mnef"];

            //Wabef
            ambit::tensor::tensor Wabef("Wabef","a,b,e,f");
            Wabef["abef"] = G_p["abef"];
            Wabef["abef"] -= t1["me"]*G_p["amef"];
            Wabef["abef"] += t1["ma"]*G_p["mbef"];
            Wabef["abef"] += 0.5*(t2["mnab"]+t1["ma"]*t1["nb"])*G_p["mnef"];

            //Wmbej
            ambit::tensor::tensor Wmbej("Wmbej","m,b,e,j");
            Wmbej["mbej"] = G_p["mbej"];
            Wmbej["mbej"] -= t1["jf"]*G_p["mbef"];
            Wmbej["mbej"] += t1["nb"]*G_p["mnej"];
            Wmbej["mbej"] -= 0.5*(t2["jnfb"]+2*t1["jf"]*t1["nb"])*G_p["mnef"];
            Wmbej["mbej"] += 0.5*t2["njfb"]*(2*G_p["mnef"]-G_p["nmef"]);

            //Wmbej abba case
            ambit::tensor::tensor W_MbeJ("WMbeJ","m,b,e,j");
            W_MbeJ["mbej"] = -G_p["bmej"];
            W_MbeJ["mbej"] -= t1["jf"]*G_p["mbfe"];
            W_MbeJ["mbej"] += t1["nb"]*G_p["nmej"];
            W_MbeJ["mbej"] += (0.5*t2["jnfb"]+t1["jf"]*t1["nb"])*G_p["nmef"];

            //Compute new t1 and t2 amplitudes
            //t1
            ambit::tensor::tensor t1n("T1 new","i,a");
            t1n["ia"] = t1["ie"]*Fae["ae"];
            t1n["ia"] -= t1["ma"]*Fmi["mi"];
            t1n["ia"] += Fme["me"]*(2*t2["imae"]-t2["miae"]);
            t1n["ia"] += t1["me"]*(2*G_p["amie"]-G_p["maie"]);
            t1n["ia"] -= t2["mnae"]*(2*G_p["mnie"]-G_p["nmie"]);
            t1n["ia"] += t2["imef"]*(2*G_p["amef"]-G_p["amfe"]);
            t1n["ia"] *= Dia["ia"];
            //t2
            ambit::tensor::tensor t2n("T2 new","i,j,a,b");
            t2n["ijab"] = G_p["ijab"];
            t2n["ijab"] += t2["ijae"]*(Fae["be"]-0.5*t1["mb"]*Fme["me"]);
            t2n["ijab"] += t2["ijeb"]*(Fae["ae"]-0.5*t1["ma"]*Fme["me"]);
            t2n["ijab"] -= t2["imab"]*(Fmi["mj"]+0.5*t1["je"]*Fme["me"]);
            t2n["ijab"] -= t2["mjab"]*(Fmi["mi"]+0.5*t1["ie"]*Fme["me"]);
            t2n["ijab"] += (t2["mnab"]+t1["ma"]*t1["nb"])*Wmnij["mnij"];
            t2n["ijab"] += (t2["ijef"]+t1["ie"]*t1["jf"])*Wabef["abef"];
            t2n["ijab"] += (t2["imae"]-t2["miae"])*Wmbej["mbej"]-t1["ie"]*t1["ma"]*G_p["mbej"];
            t2n["ijab"] += t2["imae"]*(Wmbej["mbej"]+W_MbeJ["mbej"]);
            t2n["ijab"] += (t2["mibe"]*W_MbeJ["maej"]-t1["ie"]*t1["mb"]*G_p["amej"]);
            t2n["ijab"] += (t2["mjae"]*W_MbeJ["mbei"]-t1["je"]*t1["ma"]*G_p["bmei"]);
            t2n["ijab"] += (t2["jmbe"]-t2["mjbe"])*Wmbej["maei"]-t1["je"]*t1["mb"]*G_p["maei"];
            t2n["ijab"] += t2["jmbe"]*(Wmbej["maei"]+W_MbeJ["maei"]);
            t2n["ijab"] += t1["ie"]*G_p["ejab"];
            t2n["ijab"] += t1["je"]*G_p["ieab"];
            t2n["ijab"] -= t1["ma"]*G_p["ijmb"];
            t2n["ijab"] -= t1["mb"]*G_p["ijam"];
            t2n["ijab"] *= Dijab["ijab"];

            //Compute the new CCSD energy
            double Eccn = 2*F["ia"].dot(t1n["ia"]) + G_p["ijab"].dot(2*t2n["ijab"]+2*t1n["ia"]*t1n["jb"]-t2n["jiab"]-t1n["ja"]*t1n["ib"]);



            //Test the convergence
            //Energy Difference
            double E_change = Eccn - Ecc;
            //Print Iteration Info
            std::cout << "Iteration "<< iter << " : E(CCSD) = " << Eccn << " dE = " << E_change << "\n";
            //RMS square of T1 amplitudes
            double t1_change = (t1n["ia"]-t1["ia"]).dot(t1n["ia"]-t1["ia"]);
            //RMS square of T2 amplitudes
            double t2_change = (t2n["ijab"]-t2["ijab"]).dot(t2n["ijab"]-t2["ijab"]);
            //if converged
            if(fabs(E_change)<pow(10,-E_conv) && t1_change<pow(10,-t_conv) && t2_change<pow(10,-t_conv)) {
                std::cout << "CC converged! \n\n   CCSD Correlation Energy : "<< Eccn << "\n" <<"CCSD Total Energy : " << Eccn+Escf <<"\n";
                break;
            }
            //else, update t1 and t2 amplitudes
            else {
                t1["ia"] = t1n["ia"];
                t2["ijab"] = t2n["ijab"];
            }

            //continue next iteration..


        }







    }

    ambit::util::print::finalize();

#if defined(HAVE_MPI)
    MPI::Finalize();
#endif

    return 0;
}
