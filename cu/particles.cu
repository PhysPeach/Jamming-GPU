#include "../cuh/particles.cuh"

namespace PhysPeach{
    __global__ void glo_K(double* Kpart, double* v_dev, int len){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < len) Kpart[i_global] = v_dev[i_global] * v_dev[i_global];
    }
    double K(Particles *p){
        int flip = 0;
        glo_K<<<(D*Np + NT - 1)/NT, NT>>>(p->reduction_dev[flip], p->v_dev, D*Np);
        int remain;
        for(int len = D * Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->reduction_dev[flip], p->reduction_dev[!flip], len);
        }
        double K;
        cudaMemcpy(&K, p->reduction_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        return K/(2. * (double)Np);
    }

    __global__ void glo_U(double* Upart, double *diam_dev, double *x_dev, double L, int *list_dev, int nl, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        double U1 = 0.;
        int par1[2], par2;
        int kstart[2], kend[2];
        double diam1, x1[D];
        double xij[D], rij2, diamij, r_aij;
        double Lh = 0.5 * L;

        if(i_global < np){
            par1[0] = i_global;
            kend[0] = list_dev[par1[0] * nl] / 2;
            kstart[0] = 1;
            par1[1] = np - 1 - i_global;
            kend[1] = list_dev[par1[1] * nl];
            kstart[1] = 1 + kend[1] / 2;
            
            for(int i = 0; i < 2; i++){
                diam1 = diam_dev[par1[i]];
                for(int d = 0; d < D; d++){
                    x1[d] = x_dev[d*np + par1[i]];
                }

                for(int k = kstart[i]; k <= kend[i]; k++){
                    par2 = list_dev[par1[i] * nl + k];
                    diamij = 0.5 * (diam1 + diam_dev[par2]);
                    rij2 = 0.;
                    for(int d = 0; d < D; d++){
                        xij[d] = x1[d] - x_dev[d*np + par2];
                        if (xij[d] > Lh){xij[d] -= L;}
                        if (xij[d] < -Lh){xij[d] += L;}
                        rij2 += xij[d] * xij[d];
                    }
                    if(0 < rij2 && rij2 < diamij * diamij){
                        r_aij = sqrt(rij2)/diamij;
                        U1 += 0.5 * (1 - r_aij) * (1 - r_aij);
                    }
                }
            }
            Upart[i_global] = U1;
        }
    }
    double U(Particles *p, double L, Lists* lists){
        int flip = 0;
        glo_U<<<(Np + NT - 1)/NT, NT>>>(p->reduction_dev[flip], p->diam_dev, p->x_dev, L, lists->list_dev, lists->Nl, Np);
        int remain;
        for(int len = Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->reduction_dev[flip], p->reduction_dev[!flip], len);
        }
        double U;
        cudaMemcpy(&U, p->reduction_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        return U/(double)Np;
    }

    __global__ void glo_P(double* Ppart, double *diam_dev, double* x_dev, double L, int *list_dev, int nl, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        double P1 = 0.;
        int par1[2], par2;
        int kstart[2], kend[2];
        double diam1, x1[D];
        double xij[D], rij2, diamij, rij, f_rij;
        double Lh = 0.5 * L;

        if(i_global < np){
            par1[0] = i_global;
            kend[0] = list_dev[par1[0] * nl] / 2;
            kstart[0] = 1;
            par1[1] = np - 1 - i_global;
            kend[1] = list_dev[par1[1] * nl];
            kstart[1] = 1 + kend[1] / 2;
            
            for(int i = 0; i < 2; i++){
                diam1 = diam_dev[par1[i]];
                for(int d = 0; d < D; d++){
                    x1[d] = x_dev[d*np + par1[i]];
                }
                for(int k = kstart[i]; k <= kend[i]; k++){
                    par2 = list_dev[par1[i] * nl + k];
                    diamij = 0.5 * (diam1 + diam_dev[par2]);
                    rij2 = 0.;
                    for(int d = 0; d < D; d++){
                        xij[d] = x1[d] - x_dev[d*np + par2];
                        if (xij[d] > Lh){xij[d] -= L;}
                        if (xij[d] < -Lh){xij[d] += L;}
                        rij2 += xij[d] * xij[d];
                    }
                    if(0 < rij2 && rij2 < diamij * diamij){
                        rij = sqrt(rij2);
                        f_rij = 1/(rij * diamij) - 1/(diamij * diamij);
                        P1 += 0.5 * f_rij * rij2;
                    }
                }
            }
            Ppart[i_global] = P1;
        }
    }
    double P(Particles *p, double L, Lists* lists){
        int flip = 0;
        glo_P<<<(Np + NT - 1)/NT, NT>>>(p->reduction_dev[flip], p->diam_dev, p->x_dev, L, lists->list_dev, lists->Nl, Np);
        int remain;
        for(int len = Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->reduction_dev[flip], p->reduction_dev[!flip], len);
        }
        double P;
        cudaMemcpy(&P, p->reduction_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);

        double Vol = powInt(L, D);
        return P /= Vol;
    }
    __global__ void glo_updateForces(double* f_dev, double *diam_dev, double* x_dev, double L, int *list_dev, int nl, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        int par1[2], par2;
        int kstart[2], kend[2];
        double diam1, x1[D], f1[D];
        double xij[D], rij, rij2, diamij, f_rij;
        double Lh = 0.5 * L;

        if(i_global < np){
            par1[0] = i_global;
            kend[0] = list_dev[par1[0] * nl] / 2;
            kstart[0] = 1;
            par1[1] = np - 1 - i_global;
            kend[1] = list_dev[par1[1] * nl];
            kstart[1] = 1 + kend[1] / 2;
            
            for(int i = 0; i < 2; i++){
                diam1 = diam_dev[par1[i]];
                for(int d = 0; d < D; d++){
                    x1[d] = x_dev[d*np + par1[i]];
                    f1[d] = 0.;
                }
                for(int k = kstart[i]; k <= kend[i]; k++){
                    par2 = list_dev[par1[i] * nl + k];
                    diamij = 0.5 * (diam1 + diam_dev[par2]);
                    rij2 = 0.;
                    for(int d = 0; d < D; d++){
                        xij[d] = x1[d] - x_dev[d*np + par2];
                        if (xij[d] > Lh){xij[d] -= L;}
                        if (xij[d] < -Lh){xij[d] += L;}
                        rij2 += xij[d] * xij[d];
                    }
                    if(0 < rij2 && rij2 < diamij * diamij){
                        rij = sqrt(rij2);
                        f_rij = 1/(rij * diamij) - 1/(diamij * diamij);
                        for(int d = 0; d < D; d++){
                            f1[d] += f_rij * xij[d];
                            atomicAdd(&f_dev[d*np+par2], -f_rij * xij[d]);
                        }
                    }
                }
                for(int d = 0; d < D; d++){
                    atomicAdd(&f_dev[d*np+par1[i]], f1[d]);
                }
            }
        }
    }
    void updateForces(Particles *p, double L, Lists* lists){
        fillSameNum_double<<<(D*Np + NT - 1)/NT, NT>>>(p->f_dev, 0., D*Np);
        glo_updateForces<<<(Np + NT - 1)/NT, NT>>>(p->f_dev, p->diam_dev, p->x_dev, L, lists->list_dev, lists->Nl, Np);
        return;
    }

    double powerParticles(Particles* p){
        int flip = 0;
        glo_innerProduct<<<(D * Np + NT - 1)/NT, NT>>>(p->reduction_dev[flip], p->v_dev, p->f_dev, D * Np);
        int remain;
        for(int len = D * Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->reduction_dev[flip], p->reduction_dev[!flip], len);
        }
        double power;
        cudaMemcpy(&power, p->reduction_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
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
        cudaMalloc((void**)&p->v_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->f_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->rnd_dev, D * Np * sizeof(curandState));

        cudaMalloc((void**)&p->reduction_dev[0], D * Np * sizeof(double));
        cudaMalloc((void**)&p->reduction_dev[1], D * Np * sizeof(double));

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
        cudaMalloc((void**)&p->v_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->f_dev, D * Np * sizeof(double));
        cudaMalloc((void**)&p->rnd_dev, D * Np * sizeof(curandState));

        cudaMalloc((void**)&p->reduction_dev[0], D * Np * sizeof(double));
        cudaMalloc((void**)&p->reduction_dev[1], D * Np * sizeof(double));

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
        return;
    }

    void deleteParticles(Particles *p){
        free(p->diam);
        free(p->x);
        free(p->v);
        cudaFree(p->diam_dev);
        cudaFree(p->x_dev);
        cudaFree(p->v_dev);
        cudaFree(p->f_dev);
        cudaFree(p->rnd_dev);

        cudaFree(p->reduction_dev[0]);
        cudaFree(p->reduction_dev[1]);

        return;
    }

    __global__ void vEvo(double *v_dev, double *f_dev, double dt, int len){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < len){
            v_dev[i_global] += dt * f_dev[i_global];
        }
    }
    __global__ void xEvo(double *x_dev, double *v_dev, double *f_dev, double L, double dt, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        double Lh = 0.5 * L;
        double x[D];
        if(i_global < np){
            for(int d = 0; d < D; d++){
                x[d] = x_dev[d*np+i_global] + dt * (v_dev[d*np+i_global] + 0.5 * dt * f_dev[d*np+i_global]);
                if(x[d] > Lh){x[d] -= L;}
                if(x[d] < -Lh){x[d] += L;}
                x_dev[d*np+i_global] = x[d];
            }
        }
    }
    void updateParticles(Particles *p, double L, double dt, Lists *lists){

        vEvo<<<(D*Np + NT - 1)/NT, NT>>>(p->v_dev, p->f_dev, 0.5 * dt, D*Np);
        updateForces(p, L, lists);
        vEvo<<<(D*Np + NT - 1)/NT, NT>>>(p->v_dev, p->f_dev, 0.5 * dt, D*Np);
        xEvo<<<(Np + NT - 1)/NT, NT>>>(p->x_dev, p->v_dev, p->f_dev, L, dt, Np);

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
        absolute<<<(Np + NT - 1)/NT,NT>>>(p->reduction_dev[flip], p->f_dev, Np);
        int remain;
        for(int len = Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(p->reduction_dev[flip], p->reduction_dev[!flip], len);
        }
        double fsum;
        cudaMemcpy(&fsum, p->reduction_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        if(fsum > fmax * (double)Np){
            return false;
        }
        return true;
    }
}