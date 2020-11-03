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

void record(PhysPeach::Jamming *jam, bool valid, double delta){
    std::ofstream file;
    std::ostringstream validName;
    validName << "../squeeze/valid/valid_N" << Np << "_Phi" << Phi_init << "_Dphi" << delta << "_id" << ID <<".data";
    file.open(validName.str().c_str());
    file << valid;
    file.close();

    if(valid){
        std::ostringstream squeezeName;
        squeezeName << "../squeeze/data/sq_N" << Np << "_Phi" << Phi_init << "_Dphi" << delta << "_id" << ID <<".data";
        file.open(squeezeName.str().c_str());
        file << std::setprecision(15);
        for(int par1 = 0; par1 < Np; par1++){
            file << jam->p.diam[par1] << " ";
            for(int d = 0; d < D; d++){
                file << jam->p.x[d*Np + par1] << " ";
            }
            file << std::endl;
        }
        file.close();
    }
    return;
}

int main(int argc, char** argv) {
    ID = atoi(argv[1]);
    Np = atoi(argv[2]);
    Phi_init = atof(argv[3]);

    std::cout << "-- squeeze jamming --" << std::endl;
    std::cout << "ID       : " << ID << std::endl;
    std::cout << "Np       : " << Np << std::endl;
    std::cout << "Phi_init : " << Phi_init << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;

    PhysPeach::Jamming jam;
    PhysPeach::loadJamming(&jam);
    std::cout << "Squeeze from phi = " << jam.phi << std::endl;
    std::cout << "    phi, E, P, loop:" << std::endl;
    double jammingPoint = jam.phi;
    bool valid = true;
    double delta = 0.;
    int loop;
    double Pnow = P(&jam.p, L(&jam), &jam.lists);

    if(Pnow < 1.0e-8) valid = false;
    cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    record(&jam, valid, 0.);
    if(valid){
        loop = addDphi(&jam, 1.0e-5);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
        if(Pnow < 1.0e-8) valid = false;
        cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    }
    delta += 1.0e-5;
    record(&jam, valid, delta); //1.0e-5

    if(valid){
        loop = addDphi(&jam, 4.0e-5);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
        if(Pnow < 1.0e-8) valid = false;
        cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    }
    delta += 4.0e-5;
    record(&jam, valid, delta); //5.0e-5

    if(valid){
        loop = addDphi(&jam, 5.0e-5);
        Pnow = P(&jam.p, L(&jam), &jam.lists);
        std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
        if(Pnow < 1.0e-8) valid = false;
        cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    }
    delta += 5.0e-5;
    record(&jam, valid, delta); //1.0e-4

    double dphi = 1.0e-4;

    for(int i = 0; i < 4; i++){
        if(valid){
            loop = addDphi(&jam, dphi);
            Pnow = P(&jam.p, L(&jam), &jam.lists);
            std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
            if(Pnow < 1.0e-8) valid = false;
        }
    }
    delta += 4.0e-4;
    cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    record(&jam, valid, delta); //delta = 5.0e-4

    for(int i = 0; i < 5; i++){
        if(valid){
            loop = addDphi(&jam, dphi);
            Pnow = P(&jam.p, L(&jam), &jam.lists);
            std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
            if(Pnow < 1.0e-8) valid = false;
        }
    }
    delta += 5.0e-4;
    cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    record(&jam, valid, delta); //delta = 1.0e-3

    for(int i = 0; i < 40; i++){
        if(valid){
            loop = addDphi(&jam, dphi);
            Pnow = P(&jam.p, L(&jam), &jam.lists);
            std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
            if(Pnow < 1.0e-8) valid = false;
        }
    }
    delta += 4.0e-3;
    cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    record(&jam, valid, delta); //delta = 5.0e-3

    for(int i = 0; i < 50; i++){
        if(valid){
            loop = addDphi(&jam, dphi);
            Pnow = P(&jam.p, L(&jam), &jam.lists);
            std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
            if(Pnow < 1.0e-8) valid = false;
        }
    }
    delta += 5.0e-3;
    cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
    record(&jam, valid, delta); //delta = 1.0e-2

    for(int j = 2; j <= 5; j++){
        for(int i = 0; i < 100; i++){
            if(valid){
                loop = addDphi(&jam, dphi);
                Pnow = P(&jam.p, L(&jam), &jam.lists);
                std::cout << "    " << jam.phi << ", " << U(&jam.p, L(&jam), &jam.lists) << ", " << Pnow << ", " << loop << std::endl;
                if(Pnow < 1.0e-8) valid = false;
            }
        }
        delta += 1.0e-2;
        cudaMemcpy(jam.p.diam, jam.p.diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(jam.p.x, jam.p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
        record(&jam, valid, delta);
    }
    std::cout << "finished!: phi = " << jam.phi << std::endl;

    PhysPeach::deleteJamming(&jam);

    return 0;
}