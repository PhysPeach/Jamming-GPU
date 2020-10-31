#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../cuh/MT.cuh"
#include "../cuh/conf.cuh"

#include "../cuh/cells.cuh"

namespace PhysPeach{
    struct Particles{
        //host
        double *diam;
        double packing;
        double *x;
        double *v;

        //device
        double *diam_dev;
        double *x_dev;
        double *v_dev;
        double *f_dev;
        curandState *rnd_dev;

        double *reduction_dev[2];
    };
    double K(Particles*);
    double U(Particles*, double, Lists*);
    double P(Particles*, double, Lists*);
    void updateForces(Particles*, double, Lists*);
    double powerParticles(Particles* p);
    void createParticles(Particles*);
    void createParticles(Particles*, std::ifstream*);
    void deleteParticles(Particles*);
    void updateParticles(Particles*, double, double, Lists*);
    __global__ void modifyVelocities(double*, double*, double , int);
    bool convergedFire(Particles*);
}

#endif