#include "../cuh/particles.cuh"

namespace PhysPeach{
    double powerParticles(Particles* p){
        int flip = 0;
        glo_innerProduct<<<(D * Np + NT - 1)/NT, NT>>>(p->power_dev[flip], p->v_dev, p->f_dev, D * Np);
        int remain;
        for(int len = D * Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->power_dev[flip], p->power_dev[!flip], len);
        }
        double power;
        cudaMemcpy(&power, p->power_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        return power;
    }

    __global__ void init_genrand_kernel(unsigned long long s, curandState* state){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(s, i_global,0,&state[i_global]);
    }
    __global__ void createDiam(double *diam_dev, curandState *rnd_dev, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < np){
            diam_dev[i_global] = sqrt(A * a_min * a_min / (A - 2 * a_min * a_min * curand_uniform_double(&rnd_dev[i_global])));
        }
    }
    __global__ void diamD(double *diamD_dev, double *diam_dev, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < np){
            if(D == 2){
                diamD_dev[i_global] = diam_dev[i_global] * diam_dev[i_global];
            }
            else if(D == 3){
                diamD_dev[i_global] = diam_dev[i_global] * diam_dev[i_global];
            }
        }
    }
    __global__ void createPosition(double *x_dev, curandState *rnd_dev, int uniformity, double L, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < uniformity * uniformity){
            int i2 = i_global / uniformity;
            int i1 = i_global - i2 * uniformity;
            double Lu = L / (double)uniformity;
            x_dev[i_global] = (i1 + curand_uniform_double(&rnd_dev[i_global])) * Lu - 0.5 * L;
            x_dev[np + i_global] = (i2 + curand_uniform_double(&rnd_dev[np + i_global])) * Lu - 0.5 * L;
        }
        else if(uniformity * uniformity <= i_global && i_global < np){
            x_dev[i_global] = (curand_uniform_double(&rnd_dev[i_global]) - 0.5 ) * L;
            x_dev[np + i_global] = (curand_uniform_double(&rnd_dev[i_global]) - 0.5) * L;
        }
    }
    void createParticles(Particles *p){
        int NB;
        p->packing = 0;

        p->diam = (double*)malloc(Np * sizeof(double));
        p->x = (double*)malloc(D * Np * sizeof(double));
        p->v = (double*)malloc(D * Np * sizeof(double));

        cudaMalloc((void**)&p->diam_dev, Np * sizeof(double));
        cudaMalloc((void**)&p->x_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->mem_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->v_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->f_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->rnd_dev, D * Np * sizeof(curandState));

        cudaMalloc((void**)&p->power_dev[0], D * Np * sizeof(double));
        cudaMalloc((void**)&p->power_dev[1], D * Np * sizeof(double));
        cudaMalloc((void**)&p->fabs_dev[0], Np * sizeof(double));
        cudaMalloc((void**)&p->fabs_dev[1], Np * sizeof(double));

        NB = (D * Np+NT-1)/NT;
        init_genrand_kernel<<<NB,NT>>>((unsigned long long)genrand_int32(), p->rnd_dev);

        createDiam<<<NB, NT>>>(p->diam_dev, p->rnd_dev, Np);

        //packing
        double *diamD_dev[2];
        cudaMalloc((void**)&diamD_dev[0], Np * sizeof(double));
        cudaMalloc((void**)&diamD_dev[1], Np * sizeof(double));
        bool flip = 0;
        diamD<<<NB, NT>>>(diamD_dev[flip], p->diam_dev, Np);
        cudaMemcpy(p->diam, p->diam_dev, Np * sizeof(double), cudaMemcpyDeviceToHost);
        int remain;
        for(int len = Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(diamD_dev[flip], diamD_dev[!flip], len);
        }
        cudaMemcpy(&p->packing, diamD_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        p->packing *= 0.5 * pi / D;
        cudaFree(diamD_dev[0]);
        cudaFree(diamD_dev[1]);

        double L = pow(p->packing/Phi_init, 1./(double)D);
        int uniformity = (int)(pow(Np, 1./(double)D));
        createPosition<<<NB, NT>>>(p->x_dev, p->rnd_dev, uniformity, L, Np);
        cudaMemcpy(p->mem_dev, p->x_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
        return;
    }

    void createParticles(Particles *p, std::ifstream *in){
        int NB;
        p->packing = 0;

        p->diam = (double*)malloc(Np * sizeof(double));
        p->x = (double*)malloc(D*Np * sizeof(double));
        p->v = (double*)malloc(D * Np * sizeof(double));

        cudaMalloc((void**)&p->diam_dev, Np * sizeof(double));
        cudaMalloc((void**)&p->x_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->mem_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->v_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->f_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->rnd_dev, D * Np * sizeof(curandState));

        cudaMalloc((void**)&p->power_dev[0], D * Np * sizeof(double));
        cudaMalloc((void**)&p->power_dev[1], D * Np * sizeof(double));
        cudaMalloc((void**)&p->fabs_dev[0], Np * sizeof(double));
        cudaMalloc((void**)&p->fabs_dev[1], Np * sizeof(double));

        NB = (Np+NT-1)/NT;
        init_genrand_kernel<<<NB,NT>>>((unsigned long long)genrand_int32(), p->rnd_dev);

        for (int par1 = 0; par1 < Np; par1++){
            *in >> p->diam[par1];
            for(int d = 0; d < D; d++){
                *in >> p->x[par1 + d*Np];
            }
            p->packing += powInt(p->diam[par1], D);
        }
        p->packing *= 0.5 * pi / D;
        cudaMemcpy(p->diam_dev, p->diam, Np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(p->x_dev, p->x, D * Np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(p->mem_dev, p->x, D * Np * sizeof(double), cudaMemcpyHostToDevice);
        return;
    }

    void deleteParticles(Particles *p){
        free(p->diam);
        free(p->x);
        free(p->v);
        cudaFree(p->diam_dev);
        cudaFree(p->x_dev);
        cudaFree(p->mem_dev);
        cudaFree(p->v_dev);
        cudaFree(p->f_dev);
        cudaFree(p->rnd_dev);

        cudaFree(p->power_dev[0]);
        cudaFree(p->power_dev[1]);
        cudaFree(p->fabs_dev[0]);
        cudaFree(p->fabs_dev[1]);

        return;
    }

    __global__ void modifyVelocities(double *v_dev, double *f_dev, double a, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        double v, f;
        if(i_global < np){
            v = 0.;
            f = 0.;
            for(int d = 0; d < D; d++){
                v += v_dev[d*np + i_global] * v_dev[d*np + i_global];
                f += f_dev[d*np + i_global] * f_dev[d*np + i_global];
            }
            if(f > 0.){
                v = sqrt(v);
                f = sqrt(f);
                for(int d = 0; d < D; d++){
                    v_dev[d*np + i_global] = (1-a) * v_dev[d*np + i_global] + a * v * f_dev[d*np + i_global] / f;
                }
            }else{
                v = sqrt(v);
                for(int d = 0; d < D; d++){
                    v_dev[d*np + i_global] = (1-a) * v_dev[d*np + i_global];
                }
            }
        }
    }

    bool convergedFire(Particles *p){
        double fmax = 3.0e-12;
        int flip = 0;
        absolute<<<(Np + NT - 1)/NT,NT>>>(p->fabs_dev[flip], p->f_dev, Np);
        int remain;
        for(int len = Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->fabs_dev[flip], p->fabs_dev[!flip], len);
        }
        double fsum;
        cudaMemcpy(&fsum, p->fabs_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        if(fsum > fmax * (double)Np){
            return false;
        }
        return true;
    }
}