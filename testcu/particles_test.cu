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

    void convergedFireTest(){
        Particles p;
        bool converged;

        createParticles(&p);

        converged = convergedFire(&p);
        assert(converged);

        double f = 1.0e-10 * D*Np;

        f = 1.0e-10 * D*Np;
        cudaMemcpy(&p.f_dev[0], &f, sizeof(double), cudaMemcpyHostToDevice);
        converged = convergedFire(&p);
        assert(!converged);

        f = 1.0e-12 * D*Np;
        cudaMemcpy(&p.f_dev[0], &f, sizeof(double), cudaMemcpyHostToDevice);
        converged = convergedFire(&p);
        assert(converged);

        f = 3.0e-12 * D*Np;
        cudaMemcpy(&p.f_dev[D*Np - 1], &f, sizeof(double), cudaMemcpyHostToDevice);
        converged = convergedFire(&p);
        assert(!converged);

        deleteParticles(&p);

        return;
    }

    void modifyVelocitiesTest(){
        int NB = (Np + NT - 1)/NT;
        Particles p;

        createParticles(&p);

        double s = 1.;
        cudaMemcpy(&p.v_dev[0], &s, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&p.f_dev[Np], &s, sizeof(double), cudaMemcpyHostToDevice);
        modifyVelocities<<<NB, NT>>>(p.v_dev, p.f_dev, 0.2, Np);
        cudaMemcpy(&s, &p.v_dev[0], sizeof(double), cudaMemcpyDeviceToHost);
        assert(0.79 < s && s < 0.81);
        cudaMemcpy(&s, &p.v_dev[Np], sizeof(double), cudaMemcpyDeviceToHost);
        assert(0.19 < s && s < 0.21);

        modifyVelocities<<<NB, NT>>>(p.v_dev, p.f_dev, 0.3, Np);
        cudaMemcpy(&s, &p.v_dev[0], sizeof(double), cudaMemcpyDeviceToHost);
        assert(0.55 < s && s < 0.57);
        cudaMemcpy(&s, &p.v_dev[Np], sizeof(double), cudaMemcpyDeviceToHost);
        assert(0.37 < s && s < 0.39);

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
        updateCells(&cells, L, p.x_dev);
        updateLists(&lists, &cells, L, p.x_dev);

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
        updateCellList(&cells, &lists, L, p.x_dev);
        setUpdateFreq(&cells, p.v_dev);

        double E1, E2;
        for(int i = 0; i < 10000; i++){
            if(i == 1000){
                E1 = K(&p) + U(&p, L, &lists);
            }
            updateParticles(&p, L, 0.001, &lists);
            checkUpdateCellList(&cells, &lists, L, p.x_dev, p.v_dev);
            //std::cout << i << " " << K(&p) + U(&p, L, &lists) << std::endl;
        }
        E2 = K(&p) + U(&p, L, &lists);
        assert(-0.0005 < E1 - E2 && E1 - E2 < 0.0005);
        
        deleteLists(&lists);
        deleteCells(&cells);
        deleteParticles(&p);
        return;
    }
}