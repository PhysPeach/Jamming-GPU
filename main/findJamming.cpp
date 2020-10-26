#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "../hpp/conf.hpp"
#include "../hpp/jamming.hpp"

int ID;
int Np;
double Phi_init;
double Dphi;
int main(int argc, char** argv) {
    ID = atoi(argv[1]);
    Np = atoi(argv[2]);
    Phi_init = atof(argv[3]);
    Dphi = 0.;

    std::cout << "-- find jamming --" << std::endl;
    std::cout << "ID       : " << ID << std::endl;
    std::cout << "Np       : " << Np << std::endl;
    std::cout << "Phi_init : " << Phi_init << std::endl;
    std::cout << "------------------" << std::endl << std::endl;

    PhysPeach::Jamming jam;
    PhysPeach::loadSwapMC(&jam);
    PhysPeach::fireJamming(&jam);

    double phimem;
    double *xmem;
    xmem = (double*)malloc(D*Np*sizeof(double));
    memcpy(xmem, jam.p.x, D*Np*sizeof(double));

    double Pnow = P(&jam.p, L(&jam), &jam.lists);
    double loop = 0;

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-4 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (jam.phi < 0.84 || Pnow < 1.0e-8){
        phimem = jam.phi;
        memcpy(xmem, jam.p.x, D*Np*sizeof(double));
        loop = PhysPeach::addDphi(&jam, 1.0e-4);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    memcpy(jam.p.x, xmem, D*Np*sizeof(double));
    Pnow = P(&jam.p, L(&jam), &jam.lists);

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-5 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (Pnow < 1.0e-8){
        loop = PhysPeach::addDphi(&jam, 1.0e-5);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    memcpy(xmem, jam.p.x, D*Np*sizeof(double));
    Pnow = P(&jam.p, L(&jam), &jam.lists);

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-6 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (Pnow > 1.0e-8){
        phimem = jam.phi;
        memcpy(xmem, jam.p.x, D*Np*sizeof(double));
        loop = PhysPeach::addDphi(&jam, -1.0e-6);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    memcpy(jam.p.x, xmem, D*Np*sizeof(double));
    std::cout << "-> Jamming Point: " << jam.phi << std::endl;

    std::ofstream file;

    std::ostringstream jammingName;
    jammingName << "../jammingpoint/jam_N" << Np << "_Phi" << Phi_init << "_id" << ID <<".data";
    file.open(jammingName.str().c_str());
    file << std::setprecision(15) << jam.phi;
    file.close();

    std::ostringstream posName;
    posName << "../pos/pos_N" << Np << "_Phi" << Phi_init << "_id" << ID <<".data";
    file.open(posName.str().c_str());
    file << std::setprecision(15);
    for(int par1 = 0; par1 < Np; par1++){
        file << jam.p.diam[par1] << " ";
        for(int d = 0; d < D; d++){
            file << jam.p.x[d*Np + par1] << " ";
        }
        file << std::endl;
    }
    file.close();

    PhysPeach::deleteJamming(&jam);

    return 0;
}