#ifndef PARTICLES_TEST_CUH
#define PARTICLES_TEST_CUH

#include <iostream>
#include <assert.h>

#include "../cuh/particles.cuh"

namespace PhysPeach{
    void createParticlesTest();
    void squeezePositionsTest();
    void powerParticlesTest();
    void convergedFireTest();
    void updateMemTest();
    void modifyVelocitiesTest();
    void UandPTest();
    void updateForcesTest();
    void updateParticlesTest();
}
#endif