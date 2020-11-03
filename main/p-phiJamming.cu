#include <cuda.h>

#include <iostream>
#include <iomanip>
#include <sstream>

#include "../cuh/conf.cuh"
#include "../cuh/jamming.cuh"

int ID;
int Np;
double Phi_init;
double Dphi;
int main(int argc, char** argv) {
    ID = atoi(argv[1]);
    Np = atoi(argv[2]);
    Phi_init = atof(argv[3]);
    Dphi = atof(argv[4]);

    std::cout << "--  p-phi jamming  --" << std::endl;
    std::cout << "ID       : " << ID << std::endl;
    std::cout << "Np       : " << Np << std::endl;
    std::cout << "Phi_init : " << Phi_init << std::endl;
    std::cout << "Dphi     : " << Dphi << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;

    PhysPeach::Jamming jam;
    PhysPeach::loadJamming(&jam);

    std::ofstream file;

    std::cout << "Squeeze from phi = " << jam.phi << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    double jammingPoint = jam.phi;
    double dphi;
    std::ostringstream PphiName;
    PphiName << "../p-phi/p-phi_N" << Np << "_Phi" << Phi_init << "_Dphi" << Dphi << "_id" << ID <<".data";
    file.open(PphiName.str().c_str());

    double delta = 0.;
    double Pnow;
    int loop;
    for(dphi = 1.0e-5; dphi < 1.0e-4; dphi *= 1.1){
        loop = addDphi(&jam, dphi);
        delta = jam.phi - jammingPoint;
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
        file << delta << " " << Pnow << std::endl;
        if(delta >= Dphi || Pnow <= 1.0e-8){
            break;
        }
    }
    dphi = 1.0e-4;
    double OutputAt = jam.phi - jammingPoint;
    while(delta < Dphi && Pnow > 1.0e-8){
        loop = addDphi(&jam, dphi);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
        delta = jam.phi - jammingPoint;
        if(delta >= OutputAt){
            file << delta << " " << P(&jam.p, L(&jam), &jam.lists) << std::endl;
            OutputAt = 1.1 * delta;
        }
    }
    file << delta << " " << P(&jam.p, L(&jam), &jam.lists) << std::endl;
    if(Pnow < 1.0e-8){
        std::cout << "melted !" << std::endl;
    }
    file.close();
    std::cout << "finished!: phi = " << jam.phi << std::endl;

    PhysPeach::deleteJamming(&jam);

    return 0;
}