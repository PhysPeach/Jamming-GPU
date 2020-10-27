#include "../testcuh/conf_test.cuh"

namespace PhysPeach{
    void powIntTest(){
        assert(powInt(2, 3) == 8);
        assert(powInt(3, 2) == 9);
        assert(powInt(4, 0) == 1);
        assert(powInt(2.1, 3) >= 9.26);
        assert(powInt(2.1, 3) <= 9.262);

        return;
    }
    void setZeroTest(){
        double arr[10];
        for(int arr1 = 0; arr1 < 10; arr1++){
            arr[arr1] = (double)arr1;
        }
        for(int arr1 = 0; arr1 < 10; arr1++){
            assert(arr[arr1] == (double)arr1);
        }
        setZero(arr, 10);
        for(int arr1 = 0; arr1 < 10; arr1++){
            assert(arr[arr1] == 0.);
        }
    }
}