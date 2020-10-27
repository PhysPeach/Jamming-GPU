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

    void powerParticlesTest(){
        Particles p;
        double power;

        createParticles(&p);

        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = 3.;
        }
        cudaMemcpy(p.v_dev, p.v, D * Np * sizeof(double), cudaMemcpyHostToDevice);
        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = 2.;
        }
        cudaMemcpy(p.f_dev, p.v, D * Np * sizeof(double), cudaMemcpyHostToDevice);

        power = powerParticles(&p);
        assert(5.99 * D*Np < power && power < 6.01 * D*Np);

        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = 1.;
        }
        cudaMemcpy(p.v_dev, p.v, D * Np * sizeof(double), cudaMemcpyHostToDevice);
        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = (double)par1;
        }
        cudaMemcpy(p.f_dev, p.v, D * Np * sizeof(double), cudaMemcpyHostToDevice);

        power = powerParticles(&p);
        assert((double)(D*Np * (D*Np - 1)/2) - 0.1 < power && power < (double)(D*Np * (D*Np - 1)/2) + 0.1);

        deleteParticles(&p);
        
        return;
    }
}