#ifndef CELLS_HPP
#define CELLS_HPP

#include <iostream>
#include <stdlib.h>
#include "../hpp/conf.hpp"

namespace PhysPeach{
    struct Cells{
        int *cell;
        int numOfCellsPerSide;
        int Nc;
    };
    void createCells(Cells*, double);
    void deleteCells(Cells*);
    void increaseNc(Cells*);
    void updateCells(Cells*, double, double*);

    struct Lists{
        int *list;
        int Nl;
    };
    void createLists(Lists*, Cells*);
    void deleteLists(Lists*);
    void increaseNl(Lists*);
    void updateLists(Lists*, Cells*, double, double*);
    void updateCellList(Cells*, Lists*, double, double*);
}
#endif