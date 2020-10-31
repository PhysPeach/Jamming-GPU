#include "../cuh/cells.cuh"

namespace PhysPeach{
    //cells
    void createCells(Cells *cells, double L){
        cells->numOfCellsPerSide = (int)(L/(2. * a_max));
        if(cells->numOfCellsPerSide < 3){
            cells->numOfCellsPerSide = 3;
        }
        double buf = 3.;
        cells->Nc = (int)(buf * (double)Np/ (double)powInt(cells->numOfCellsPerSide, D));

        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        cudaMalloc((void**)&cells->cell_dev, NoC * sizeof(int));

        cudaMalloc((void**)&cells->vmax_dev[0], D * Np * sizeof(double));
        cudaMalloc((void**)&cells->vmax_dev[1], D * Np * sizeof(double));
        cells->updateFreq = 25;
        return;
    }

    void deleteCells(Cells *cells){
        cudaFree(cells->vmax_dev[0]);
        cudaFree(cells->vmax_dev[1]);
        cudaFree(cells->cell_dev);
        return;
    }

    void increaseNc(Cells *cells){
        cells->Nc = (int)(1.4 * cells->Nc);
        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        cudaFree(cells->cell_dev);
        cudaMalloc((void**)&cells->cell_dev, NoC * sizeof(int));
        return;
    }

    __global__ void glo_putParticlesIntoCells(int *cell_dev, int numOfCellsPerSide, int nc, double *x_dev, double L, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        double Lc = L/(double)numOfCellsPerSide;
        double Lh = 0.5 * L;

        int c[D];
        int counter;
        if(i_global < np){
            c[0] = (x_dev[i_global] + Lh)/Lc;
            c[1] = (x_dev[np+i_global] + Lh)/Lc;
            if(c[0] >= numOfCellsPerSide){c[0] -= numOfCellsPerSide;}
            if(c[0] < 0){c[0] += numOfCellsPerSide;}
            if(c[1] >= numOfCellsPerSide){c[1] -= numOfCellsPerSide;}
            if(c[1] < 0){c[1] += numOfCellsPerSide;}

            counter = 1 + atomicAdd(&cell_dev[(c[0]*numOfCellsPerSide+c[1])*nc], 1);
            //if(counter >= nc){
            //    return false; //bug
            //}
            cell_dev[(c[0]*numOfCellsPerSide+c[1])*nc + counter] = i_global;
        }
    }
    bool putParticlesIntoCells(Cells *cells, double L, double* x_dev){
        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        fillSameNum_int<<<(NoC + NT - 1)/NT, NT>>>(cells->cell_dev, 0, NoC);
        glo_putParticlesIntoCells<<<(Np + NT - 1)/NT, NT>>>(cells->cell_dev, cells->numOfCellsPerSide, cells->Nc, x_dev, L, Np);
        return true;
    }

    void updateCells(Cells *cells, double L, double* x_dev){
        bool success = false;
        success = putParticlesIntoCells(cells, L, x_dev);
        while (!success){
            std::cout << "hello" << std::endl;
            increaseNc(cells);
            success = putParticlesIntoCells(cells, L, x_dev);
        }
        return;
    }

    //lists
    void createLists(Lists *lists, Cells *cells){
        lists->Nl = (int)(3.2 * (double)cells->Nc);

        int NoL = lists->Nl * Np;
        cudaMalloc((void**)&lists->list_dev, NoL * sizeof(int));
        return;
    }
    
    void deleteLists(Lists *lists){
        cudaFree(lists->list_dev);
        return;
    }

    void increaseNl(Lists *lists){
        lists->Nl = (int)(1.4 * lists->Nl);
        int NoL = lists->Nl * Np;
        cudaFree(lists->list_dev);
        cudaMalloc((void**)&lists->list_dev, NoL * sizeof(int));
        return;
    }

    __device__ int fix(int i, int M) {
	    if (i < 0) i += M;
	    if (i >= M) i -= M;

	    return i;
    }
    __global__ void glo_putParticlesIntoLists(int *list_dev, int nl, int *cell_dev, int numOfCellsPerSide, int nc, double *x_dev, double L, int np){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        int c1[D], c2[D], c3[D];
        int numOfParticlesInCell;
        double Lc = L/(double)numOfCellsPerSide;
        double Lh = 0.5 * L;

        double x1[D];
        int par2;
        double dx[D], dr;

        int counter;
        if(i_global < np){
            for(int d = 0; d < D; d++){
                x1[d] = x_dev[d*np+i_global];
            }
            c1[0] = (x1[0] + Lh)/Lc;
            c1[1] = (x1[1] + Lh)/Lc;
            if(c1[0] >= numOfCellsPerSide){c1[0] -= numOfCellsPerSide;}
            if(c1[0] < 0){c1[0] += numOfCellsPerSide;}
            if(c1[1] >= numOfCellsPerSide){c1[1] -= numOfCellsPerSide;}
            if(c1[1] < 0){c1[0] += numOfCellsPerSide;}
    
            for(c2[0] = c1[0]-1; c2[0] <= c1[0]+1; c2[0]++){
                c3[0] = fix(c2[0], numOfCellsPerSide);
                for(c2[1]= c1[1]-1; c2[1] <= c1[1]+1; c2[1]++){
                    c3[1] = fix(c2[1], numOfCellsPerSide);
                    numOfParticlesInCell = cell_dev[(c3[0]*numOfCellsPerSide+c3[1])*nc];
                    for(int k = 1; k <= numOfParticlesInCell;k++){
                        par2 = cell_dev[(c3[0]*numOfCellsPerSide+c3[1])*nc + k];
                        if(par2 > i_global){
                            dr = 0.;
                            for(int d = 0; d < D; d++){
                                dx[d] = x1[d] - x_dev[d*np+par2];
                                if(dx[d] < -Lh) dx[d] += L;
                                if(dx[d] > Lh) dx[d] -= L;
                                dr += dx[d] * dx[d];
                            }
                            if(dr < 4 * a_max * a_max){
                                counter = 1 + atomicAdd(&list_dev[i_global*nl], 1);
                                //if(list_dev[par1*nl] >= nl){
                                //    return false;
                                //}
                                list_dev[i_global*nl + counter] = par2;
                            }
                        }
                    }
                }
            }
        }
    }
    bool putParticlesIntoLists(Lists *lists, Cells *cells, double L, double* x_dev){
        int NoL = lists->Nl * Np;
        fillSameNum_int<<<(NoL + NT - 1)/NT, NT>>>(lists->list_dev, 0, NoL);
        glo_putParticlesIntoLists<<<(Np + NT - 1)/NT, NT>>>(lists->list_dev, lists->Nl, cells->cell_dev, cells->numOfCellsPerSide, cells->Nc, x_dev, L, Np);
        return true;
    }

    void updateLists(Lists *lists, Cells *cells, double L, double* x_dev){
        bool success = false;
        success = putParticlesIntoLists(lists, cells, L, x_dev);
        while (!success){
            std::cout << "hi" << std::endl;
            increaseNl(lists);
            success = putParticlesIntoLists(lists, cells, L, x_dev);
        }
        return;
    }

    void updateCellList(Cells *cells, Lists *lists, double L, double *x_dev){
        updateCells(cells, L, x_dev);
        updateLists(lists, cells, L, x_dev);
        return;
    }

    void setUpdateFreq(Cells *cells, double *v_dev){
        double vmax;
        int flip = 0;

        cudaMemcpy(cells->vmax_dev[flip], v_dev, D * Np * sizeof(double), cudaMemcpyDeviceToDevice);
        int remain;
        for(int len = D * Np; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            maxReduction<<<remain, NT>>>(cells->vmax_dev[flip], cells->vmax_dev[!flip], len);
        }
        cudaMemcpy(&vmax, cells->vmax_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        cells->updateFreq = 25;
        if(vmax != 0.){
            if(vmax < 0){
                vmax *= -1.;
            }
            cells->updateFreq = (int)(0.5 * a_max/(vmax * dt_max));
        }
        if(cells->updateFreq > 25){
            cells->updateFreq = 25;
        }
        return;
    }

    void checkUpdateCellList(Cells *cells, Lists *lists, double L, double *x_dev, double *v_dev){
        static uint counter = 0;
        counter++;
        if(counter >= cells->updateFreq){
            updateCellList(cells, lists, L, x_dev);
            setUpdateFreq(cells, v_dev);
            counter = 0;
        }
        return;
    }
}