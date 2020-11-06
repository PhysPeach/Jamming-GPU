#include <curand.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include "../cuh/MT.cuh"
#include "../cuh/conf.cuh"
#include "../cuh/cells.cuh"
#include "../cuh/particles.cuh"

double measureTime(){
    static int active = 0;
    static time_t s;
    static suseconds_t us;
    double ms;
    struct timeval tv;
    struct timezone tz;

    if(active){
        gettimeofday(&tv,&tz);
        ms = 1.0e+3 * (tv.tv_sec - s) + 1.0e-3 * (tv.tv_usec - us);
        active = 0;
    }
    else{
        ms = 0.0;
        active = 1;
        gettimeofday(&tv,&tz);
        s = tv.tv_sec;
        us = tv.tv_usec;
    }
    return ms;
}

int ID;
int Np;
double Phi_init;
double Dphi;

int main(){

    ID = 0;
    Np = 16384;
    Phi_init = 0.8;
    Dphi = 0.05;

    std::cout << "--  p-phi jamming  --" << std::endl;
    std::cout << "ID       : " << ID << std::endl;
    std::cout << "Np       : " << Np << std::endl;
    std::cout << "Phi_init : " << Phi_init << std::endl;
    std::cout << "Dphi     : " << Dphi << std::endl;
    std::cout << "---------------------" << std::endl << std::endl;

    PhysPeach::Particles p;
    PhysPeach::Cells cells;
    PhysPeach::Lists lists;

    PhysPeach::createParticles(&p);
    double L = pow(p.packing/Phi_init, 1./(double)D);
    PhysPeach::createCells(&cells, L);
    PhysPeach::createLists(&lists, &cells);
    PhysPeach::updateCellList(&cells, &lists, L, p.x_dev);
    PhysPeach::setUpdateFreq(&cells, p.v_dev);

    double E;
    double result;
    measureTime();
    for(int i = 0; i < 10000; i++){
        PhysPeach::updateParticles(&p, L, 0.001, &lists);
        PhysPeach::checkUpdateCellList(&cells, &lists, L, p.x_dev, p.v_dev);
    }
    E = PhysPeach::K(&p) + PhysPeach::U(&p, L, &lists);
    result = measureTime();
    std::cout << "1loop Time: " << result / 10000. << "ms" << std::endl;
    
    PhysPeach::deleteLists(&lists);
    PhysPeach::deleteCells(&cells);
    PhysPeach::deleteParticles(&p);
    return 0;
}