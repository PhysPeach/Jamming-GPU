#ifndef JAMMING_CUH
#define JAMMING_CUH

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include "../cuh/conf.cuh"
#include "../cuh/particles.cuh"
#include "../cuh/cells.cuh"

namespace PhysPeach{
    struct Jamming {
        double phi;
        Particles p;
        Cells cells;
        Lists lists;
    };
    double L(Jamming*);
    void createJamming(Jamming*);
    void loadSwapMC(Jamming*);
    void loadJamming(Jamming*);
    void deleteJamming(Jamming*);
    int fireJamming(Jamming*);
    int addDphi(Jamming*, double);
}

#endif