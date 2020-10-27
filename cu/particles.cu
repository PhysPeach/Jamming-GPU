#include "../cuh/particles.cuh"

namespace PhysPeach{
    double K(Particles *p){
        double K = 0.;
        for(int par1 = 0; par1 < D*Np; par1++){
            K += p->v[par1] * p->v[par1];
        }
        return K/(2. * (double)Np);
    }
    double U(Particles *p, double L, Lists* lists){
        double U = 0.;

        double diam1, x1[D];
        int par2;
        double xij[D], rij2, diamij, r_aij;
        int cell1, cell2, cell3;
        int list;
        double Lh = 0.5 * L;
        for(int par1 = 0; par1 < Np; par1++){
            diam1 = p->diam[par1];
            for(int d = 0; d < D; d++){
                x1[d] = p->x[d*Np + par1];
            }
            for(int k = 1; k<= lists->list[par1*lists->Nl]; k++){
                par2 = lists->list[par1*lists->Nl + k];
                diamij = 0.5 * (diam1 + p->diam[par2]);
                rij2 = 0.;
                for(int d = 0; d < D; d++){
                    xij[d] = x1[d] - p->x[d*Np + par2];
                    if (xij[d] > Lh){xij[d] -= L;}
                    if (xij[d] < -Lh){xij[d] += L;}
                    rij2 += xij[d] * xij[d];
                }
                if(0 < rij2 && rij2 < diamij * diamij){
                    r_aij = sqrt(rij2)/diamij;
                    U += 0.5 * (1 - r_aij) * (1 - r_aij);
                }
            }
        }
        return U/(double)Np;
    }

    double P(Particles *p, double L, Lists* lists){
        double P = 0.;

        double diam1, x1[D];
        int par2;
        double xij[D], rij, rij2, diamij, f_rij;
        int cell1, cell2, cell3;
        int list;
        double Lh = 0.5 * L;
        for(int par1 = 0; par1 < Np; par1++){
            diam1 = p->diam[par1];
            for(int d = 0; d < D; d++){
                x1[d] = p->x[d*Np + par1];
            }
            for(int k = 1; k<= lists->list[par1*lists->Nl]; k++){
                par2 = lists->list[par1*lists->Nl + k];
                diamij = 0.5 * (diam1 + p->diam[par2]);
                rij2 = 0.;
                for(int d = 0; d < D; d++){
                    xij[d] = x1[d] - p->x[d*Np + par2];
                    if (xij[d] > Lh){xij[d] -= L;}
                    if (xij[d] < -Lh){xij[d] += L;}
                    rij2 += xij[d] * xij[d];
                }
                if(0 < rij2 && rij2 < diamij * diamij){
                    rij = sqrt(rij2);
                    f_rij = 1/(rij * diamij) - 1/(diamij * diamij);
                    P += 0.5 * f_rij * rij2;
                }
            }
        }
        double Vol = powInt(L, D);
        return P /= Vol;
    }
    
    void updateForces(Particles *p, double L, Lists* lists){
        double diam1, x1[D], f1[D];
        int par2;
        double xij[D], rij, rij2, diamij, f_rij;
        int cell1, cell2, cell3;
        int list;
        double Lh = 0.5 * L;

        setZero(p->f, D*Np);
        for(int par1 = 0; par1 < Np; par1++){
            diam1 = p->diam[par1];
            for(int d = 0; d < D; d++){
                x1[d] = p->x[d*Np + par1];
                f1[d] = 0.;
            }
            for(int k = 1; k<= lists->list[par1*lists->Nl]; k++){
                par2 = lists->list[par1*lists->Nl + k];
                diamij = 0.5 * (diam1 + p->diam[par2]);
                rij2 = 0.;
                for(int d = 0; d < D; d++){
                    xij[d] = x1[d] - p->x[d*Np + par2];
                    if (xij[d] > Lh){xij[d] -= L;}
                    if (xij[d] < -Lh){xij[d] += L;}
                    rij2 += xij[d] * xij[d];
                }
                if(0 < rij2 && rij2 < diamij * diamij){
                    rij = sqrt(rij2);
                    f_rij = 1/(rij * diamij) - 1/(diamij * diamij);
                    for(int d = 0; d < D; d++){
                        f1[d] += f_rij * xij[d];
                        p->f[par2+d*Np] -= f_rij * xij[d];
                    }
                }
            }
            for(int d = 0; d < D; d++){
                p->f[d*Np + par1] += f1[d];
            }
        }
        return;
    }

    double powerParticles(Particles* p){
        double power = 0.;
        for(int par1 = 0; par1 < D*Np; par1++){
            power += p->v[par1] * p->f[par1];
        }
        return power;
    }

    void createParticles(Particles *p){
        p->packing = 0;

        p->diam = (double*)malloc(Np * sizeof(double));
        p->x = (double*)malloc(D*Np * sizeof(double));
        p->mem = (double*)malloc(D * Np * sizeof(double));
        p->v = (double*)malloc(D * Np * sizeof(double));
        p->f = (double*)malloc(D * Np * sizeof(double));

        for (int par1 = 0; par1 < Np; par1++){
            p->diam[par1] = sqrt(A * a_min * a_min / (A - 2 * a_min * a_min * genrand_real1()));
            p->packing += powInt(p->diam[par1], D);
        }
        p->packing *= 0.5 * pi / D;

        double L = pow(p->packing/Phi_init, 1./(double)D);
        int uniformity = (int)(pow(Np, 1./(double)D));
        double Lu = L / (double)uniformity;

        for(int i1 = 0; i1 < uniformity; i1++){
            for(int i2 = 0; i2 < uniformity; i2++){
                if(D == 2){
                    p->x[i1*uniformity+i2] = (i1 + genrand_real1()) * Lu - 0.5 * L;
                    p->x[Np+i1*uniformity+i2] = (i2 + genrand_real1()) * Lu - 0.5 * L;
                }
                else if(D == 3){
                    for(int i3 = 0; i3 < uniformity; i3++){
                        p->x[(i1*uniformity+i2)*uniformity+i3] = (i1 + genrand_real1()) * Lu - 0.5 * L;
                        p->x[Np+(i1*uniformity+i2)*uniformity+i3] = (i2 + genrand_real1()) * Lu - 0.5 * L;
                        p->x[2*Np+(i1*uniformity+i2)*uniformity+i3] = (i3 + genrand_real1()) * Lu - 0.5 * L;
                    }
                }
                else{
                    std::cout << "dimention err" << std::endl;
                    exit(1);
                }
            }
        }
        for(int m = powInt(uniformity, D); m < Np; m++){
            p->x[m] = (genrand_real1() - 0.5 )* L;
            p->x[Np+m] = (genrand_real1() - 0.5) * L;
            if(D == 3){
                p->x[2*Np+m] = (genrand_real1() - 0.5) * L;
            }
        }
        memcpy(p->mem, p->x, D * Np * sizeof(double));

        for (int par1 = 0; par1 < D*Np; par1++){
            p->v[par1] = 0.;
            p->f[par1] = 0.;
        }
        return;
    }

    void createParticles(Particles *p, std::ifstream *in){
        p->packing = 0;

        p->diam = (double*)malloc(Np * sizeof(double));
        p->x = (double*)malloc(D*Np * sizeof(double));
        p->mem = (double*)malloc(D * Np * sizeof(double));
        p->v = (double*)malloc(D * Np * sizeof(double));
        p->f = (double*)malloc(D * Np * sizeof(double));

        for (int par1 = 0; par1 < Np; par1++){
            *in >> p->diam[par1];
            for(int d = 0; d < D; d++){
                *in >> p->x[par1 + d*Np];
                p->v[par1 + d*Np] = 0;
                p->f[par1 + d*Np] = 0;
            }
            p->packing += powInt(p->diam[par1], D);
        }
        p->packing *= 0.5 * pi / D;

        memcpy(p->mem, p->x, D * Np * sizeof(double));
        return;
    }

    void deleteParticles(Particles *p){
        free(p->diam);
        free(p->x);
        free(p->mem);
        free(p->v);
        free(p->f);
        return;
    }

    bool updateParticles(Particles *p, double L, double dt, Lists *lists){
        double Lh = 0.5 * L;

        for(int par1 = 0; par1 < D*Np; par1++){
            p->v[par1] += 0.5 * dt * p->f[par1];
        }
        updateForces(p, L, lists);
        for(int par1 = 0; par1 < Np; par1++){
            for(int d = 0; d < D; d++){
                p->v[d*Np+par1] += 0.5 * dt * p->f[d*Np+par1];
                p->x[d*Np+par1] += dt * (p->v[d*Np+par1] + 0.5 * dt * p->f[d*Np+par1]);
                if(p->x[d*Np+par1] > Lh){p->x[d*Np+par1] -= L;}
                if(p->x[d*Np+par1] < -Lh){p->x[d*Np+par1] += L;}
            }
        }
        return updateMem(p, L);
    }

    bool updateMem(Particles* p, double L){
        double Lh = 0.5 * L;
        double dx[D];
        double frag = 0.25 * a_max * a_max;
        //fix mem_x by gamma
        /*for(int par1 = 0; par1 < Np; par1++){
            p->mem[par1] += (p->gamma - p->gammaMem) * L;
        }
        p->gammaMem = p->gamma;*/
        for(int par1 = 0 ; par1 < Np; par1++){
            for(int d = 0; d < D; d++){
                dx[d] = p->x[d*Np+par1] - p->mem[d*Np+par1];
            }
            for(int d = D-1; d >= 0; d--){
                if(dx[d] > Lh){dx[d] -= L;}
                if(dx[d] < -Lh){dx[d] += L;}
                if(dx[d] > frag){
                    memcpy(p->mem, p->x, D * Np * sizeof(double));
                    return true;
                }
            }
        }
        return false;
    }

    void squeezePositions(Particles *p, double a){
            for(int par1 = 0; par1 < D * Np; par1++){
            p->x[par1] *= a;
            p->mem[par1] *= a;
        }
        return;
    }

    void modifyVelocities(Particles* p, double a){
        double v, f;
        for(int par1 = 0; par1 < Np; par1++){
            v = 0.;
            f = 0.;
            for(int d = 0; d < D; d++){
                v += p->v[d*Np+par1] * p->v[d*Np+par1];
                f += p->f[d*Np+par1] * p->f[d*Np+par1];
            }
            if(f > 0.){
                v = sqrt(v);
                f = sqrt(f);
                for(int d = 0; d < D; d++){
                    p->v[d*Np+par1] = (1-a) * p->v[d*Np+par1] + a * v * p->f[d*Np+par1] / f;
                }
            }else{
                v = sqrt(v);
                for(int d = 0; d < D; d++){
                    p->v[d*Np+par1] = (1-a) * p->v[d*Np+par1];
                }
            }
        }
        return;
    }

    bool convergedFire(Particles *p){
        double fsum = 0.;
        double fmax = 3.0e-12;
        double f2;
        for(int par1 = 0; par1 < Np; par1++){
            f2 = 0.;
            for(int d = 0; d < D; d++){
                f2 += p->f[par1+d*Np] * p->f[par1+d*Np];
            }
            fsum += sqrt(f2);
        }
        if(fsum > fmax * Np){
            return false;
        }
        return true;
    }
}