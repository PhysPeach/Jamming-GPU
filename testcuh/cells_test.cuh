#ifndef CELLS_TEST_CUH
#define CELLS_TEST_CUH

#include <iostream>
#include <assert.h>

#include "../cuh/particles.cuh"
#include "../cuh/cells.cuh"

namespace PhysPeach{
    //cellsTest
    void createCellsTest();
    void increaseNcTest();
    void updateCellsTest();

    //listsTest
    void createListsTest();
    void increaseNlTest();
    void updateListsTest();

    //cellListTest
    void setUpdateFreqTest();
}
#endif