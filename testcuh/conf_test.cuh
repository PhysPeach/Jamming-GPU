#ifndef CONF_TEST_CUH
#define CONF_TEST_CUH

#include <iostream>
#include <assert.h>

#include "../cuh/conf.cuh"

namespace PhysPeach{
    void powIntTest();
    void setZeroTest();
    void addReductionTest();
    void multipliedTest();
    void glo_innerProductTest();
}
#endif