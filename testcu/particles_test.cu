#include "../testcuh/particles_test.cuh"
#include <fstream>

namespace PhysPeach{
    void createParticlesTest(){

        Particles p;
        createParticles(&p);
        cudaMemcpy(p.x, p.x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);

        double diamav = 0.;
        double xav[D];
        for(int d = 0; d < D; d++){
            xav[d] = 0.;
        }
        for(int par1 = 0; par1 < Np; par1++){
            diamav += p.diam[par1];
            for(int d = 0; d < D; d++){
                xav[d] += p.x[d*Np+par1];
            }
        }

        diamav /= Np;
        assert(0.99 < diamav && diamav < 1.01);

        double L = pow(p.packing/Phi_init, 1./(double)D);
        for(int d = 0; d < D; d++){
            xav[d] /= Np * L;
            assert(-0.01 < xav[d] && xav[d] < 0.01);
        }

        double *mem;
        mem = (double*)malloc(D * Np * sizeof(double));
        cudaMemcpy(mem, p.mem_dev, D * Np * sizeof(double), cudaMemcpyDeviceToHost);
        for(int par1 = 0; par1 < D*Np; par1++){
            assert(p.x[par1] == mem[par1]);
        }
        free(mem);
        deleteParticles(&p);
        return;
    }
}