#ifndef CELLS_CUH
#define CELLS_CUH

#include <iostream>
#include <stdlib.h>
#include "../cuh/conf.cuh"

namespace PhysPeach{
    struct Cells{
        int numOfCellsPerSide;
        int Nc;
        int updateFreq;

        int *cell_dev;
        double *vmax_dev[2];

    };
    void createCells(Cells*, double);
    void deleteCells(Cells*);
    void increaseNc(Cells*);
    void updateCells(Cells*, double, double*);

    struct Lists{
        int *list_dev;
        int Nl;
    };
    void createLists(Lists*, Cells*);
    void deleteLists(Lists*);
    void increaseNl(Lists*);
    void updateLists(Lists*, Cells*, double, double*);

    void updateCellList(Cells*, Lists*, double, double*);
    void setUpdateFreq(Cells*, double*);
    void checkUpdateCellList(Cells*, Lists*, double, double* , double*);
}
#endif