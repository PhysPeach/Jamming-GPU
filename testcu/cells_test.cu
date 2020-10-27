#include "../testcuh/cells_test.cuh"

namespace PhysPeach{
    //cellsTest
    void createCellsTest(){

        Cells cells;
        createCells(&cells, 3.);
        assert(cells.numOfCellsPerSide == 3);
        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(3, D)));
        deleteCells(&cells);

        createCells(&cells, 40.);
        assert(cells.numOfCellsPerSide == 12);
        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(12, D)));
        deleteCells(&cells);

        return;
    }

    void increaseNcTest(){
        Cells cells;
        createCells(&cells, 40.);

        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(12, D)));
        increaseNc(&cells);
        assert(cells.Nc == (int)(1.4 * (int)(3. * (double)Np/ (double)powInt(12, D))));
        deleteCells(&cells);

        return;
    }
    //listsTest
}