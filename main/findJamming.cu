#include <cuda.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <time.h>

#include "../cuh/conf.cuh"
#include "../cuh/jamming.cuh"

int ID;
int Np;
double Phi_init;
double Dphi;
int main(int argc, char** argv) {

    ID = atoi(argv[1]);
    init_genrand((unsigned long)time(NULL)*ID + ID);
    Np = atoi(argv[2]);
    Phi_init = atof(argv[3]);
    Dphi = 0.;

    std::cout << "-- find jamming --" << std::endl;
    std::cout << "ID       : " << ID << std::endl;
    std::cout << "Np       : " << Np << std::endl;
    std::cout << "Phi_init : " << Phi_init << std::endl;
    std::cout << "------------------" << std::endl << std::endl;

    PhysPeach::Jamming jam;
    if(Phi_init * Phi_init > 0.0001){
        PhysPeach::loadSwapMC(&jam);
    }else{
        Phi_init = 0.82;
        PhysPeach::createJamming(&jam);
        Phi_init = 0.;
    }
    PhysPeach::fireJamming(&jam);

    double phimem;
    double *xmem_dev;
    cudaMalloc((void**)&xmem_dev, D * Np * sizeof(double));
    cudaMemcpy(xmem_dev, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);

    double Pnow = P(&jam.p, L(&jam), &jam.lists);
    double loop = 0;

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-4 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (jam.phi < 0.84 || Pnow < 1.0e-8){
        phimem = jam.phi;
        cudaMemcpy(xmem_dev, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
        loop = PhysPeach::addDphi(&jam, 1.0e-4);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    cudaMemcpy(jam.p.x_dev, xmem_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
    Pnow = P(&jam.p, L(&jam), &jam.lists);

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-5 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (Pnow < 1.0e-8){
        loop = PhysPeach::addDphi(&jam, 1.0e-5);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    cudaMemcpy(xmem_dev, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
    Pnow = P(&jam.p, L(&jam), &jam.lists);

    std::cout << "    Squeeze from phi = " << jam.phi << " by dphi = " << 1.0e-6 << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    while (Pnow > 1.0e-8){
        phimem = jam.phi;
        cudaMemcpy(xmem_dev, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
        loop = PhysPeach::addDphi(&jam, -1.0e-6);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
    }
    jam.phi = phimem;
    cudaMemcpy(jam.p.x_dev, xmem_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
    std::cout << "-> Jamming Point: " << jam.phi << std::endl;
    cudaFree(xmem_dev);

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
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
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