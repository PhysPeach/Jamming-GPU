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
        return;
    }

    void deleteCells(Cells *cells){
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
    
}