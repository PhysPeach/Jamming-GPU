#include "../testcuh/particles_test.cuh"
#include <fstream>

namespace PhysPeach{
    void createParticlesTest(){

        Particles p;
        createParticles(&p);

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

        for(int par1 = 0; par1 < D*Np; par1++){
            assert(p.x[par1] == p.mem[par1]);
        }
        deleteParticles(&p);
        return;
    }

    void squeezePositionsTest(){
        Particles p;
        createParticles(&p);
        for(int par1 = 0; par1 < D*Np; par1++){
            p.x[par1] = 1.;
            p.mem[par1] = 1.;
        }
        squeezePositions(&p, 0.99);
        for(int par1 = 0; par1 < D*Np; par1++){
            assert(p.x[par1] == 1. * 0.99);
            assert(p.mem[par1] == 1. * 0.99);
        }
        deleteParticles(&p);

        return;
    }

    void powerParticlesTest(){
        Particles p;
        double power;

        createParticles(&p);
        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = 2.;
            p.f[par1] = 3.;
        }
        power = powerParticles(&p);
        assert(5.99 * D*Np < power && power < 6.01 * D*Np);

        for(int par1 = 0; par1 < D*Np; par1++){
            p.v[par1] = par1;
            p.f[par1] = 1.;
        }
        power = powerParticles(&p);
        assert((double)(D*Np * (D*Np - 1)/2) - 0.1 < power && power < (double)(D*Np * (D*Np - 1)/2) + 0.1);

        deleteParticles(&p);
        
        return;
    }

    void convergedFireTest(){
        Particles p;
        bool converged;

        createParticles(&p);

        converged = convergedFire(&p);
        assert(converged);

        p.f[0] = 1.0e-10 * D*Np;
        converged = convergedFire(&p);
        assert(!converged);

        p.f[0] = 1.0e-12 * D*Np;
        converged = convergedFire(&p);
        assert(converged);

        p.f[D*Np - 1] = 3.0e-12 * D*Np;
        converged = convergedFire(&p);
        assert(!converged);

        deleteParticles(&p);

        return;
    }

    void updateMemTest(){
        Particles p;
        bool updated;

        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);

        updated = updateMem(&p, L);
        assert(!updated);

        p.x[0] += 0.3 * a_max;
        updated = updateMem(&p, L);
        assert(!updated);
        assert(p.x[0] != p.mem[0]);

        p.x[1] += 0.3 * a_max;
        updated = updateMem(&p, L);
        assert(!updated);
        assert(p.x[1] != p.mem[1]);

        p.x[0] += 0.21 * a_max;
        updated = updateMem(&p, L);
        assert(updated);
        assert(p.x[0] == p.mem[0]);
        assert(p.x[1] == p.mem[1]);

        deleteParticles(&p);
        return;
    }

    void modifyVelocitiesTest(){
        Particles p;

        createParticles(&p);

        p.v[0] = 1.;
        p.f[Np] = 1.;
        modifyVelocities(&p, 0.2);
        assert(0.79 < p.v[0] && p.v[0] < 0.81);
        assert(0.19 < p.v[Np] && p.v[Np] < 0.21);

        p.v[1] = 0.;
        p.v[Np+1] = 2.;
        p.f[1] = 1.3;
        p.f[Np+1] = 0.;
        modifyVelocities(&p, 0.3);
        assert(0.55 < p.v[0] && p.v[0] < 0.57);
        assert(0.37 < p.v[Np] && p.v[Np] < 0.39);
        assert(0.59 < p.v[1] && p.v[1] < 0.61);
        assert(1.39 < p.v[Np+1] && p.v[Np+1] < 1.41);

        deleteParticles(&p);
        return;
    }

    void updateForcesTest(){
        Particles p;
        Cells cells;
        Lists lists;

        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);
        createCells(&cells, L);
        createLists(&lists, &cells);
        updateCells(&cells, L, p.x);
        updateLists(&lists, &cells, L, p.x);

        updateForces(&p, L, &lists);

        deleteLists(&lists);
        deleteCells(&cells);
        deleteParticles(&p);

        return;
    }

    void UandPTest(){
        Particles p;
        Cells cells;
        Lists lists;

        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);
        createCells(&cells, L);
        createLists(&lists, &cells);
        updateCells(&cells, L, p.x);
        updateLists(&lists, &cells, L, p.x);

        //std::cout << "U: " << U(&p, L, &lists) << std::endl;
        //std::cout << "P: " << P(&p, L, &lists) << std::endl;

        deleteLists(&lists);
        deleteCells(&cells);
        deleteParticles(&p);

        return;
    }

    void updateParticlesTest(){
        Particles p;
        Cells cells;
        Lists lists;

        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);
        createCells(&cells, L);
        createLists(&lists, &cells);
        updateCells(&cells, L, p.x);
        updateLists(&lists, &cells, L, p.x);

        double updatecelllist;
        double E1, E2;
        for(int i = 0; i < 10000; i++){
            if(i == 1000){
                E1 = K(&p) + U(&p, L, &lists);
            }
            updatecelllist = updateParticles(&p, L, 0.001, &lists);
            if(updatecelllist){
                updateCells(&cells, L, p.x);
                updateLists(&lists, &cells, L, p.x);
            }
        }
        E2 = K(&p) + U(&p, L, &lists);
        assert(-0.005 < E1 - E2 && E1 - E2 < 0.005);

        deleteLists(&lists);
        deleteCells(&cells);
        deleteParticles(&p);
        return;
    }
}