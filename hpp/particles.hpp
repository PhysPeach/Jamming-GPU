#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../hpp/MT.hpp"
#include "../hpp/conf.hpp"

#include "../hpp/cells.hpp"

namespace PhysPeach{
    struct Particles{
        double *diam;
        double packing;
        double *x;
        double *mem;
        double *v;
        double *f;
    };
    double K(Particles*);
    double U(Particles*, double, Lists*);
    double P(Particles*, double, Lists*);
    void updateForces(Particles*, double, Lists*);
    double powerParticles(Particles* p);
    void createParticles(Particles*);
    void createParticles(Particles*, std::ifstream*);
    void deleteParticles(Particles*);
    bool updateParticles(Particles*, double, double, Lists*);
    bool updateMem(Particles*, double);
    void squeezePositions(Particles*, double);
    void modifyVelocities(Particles*, double);
    bool convergedFire(Particles*);
}

#endif