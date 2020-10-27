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
    //lists
}