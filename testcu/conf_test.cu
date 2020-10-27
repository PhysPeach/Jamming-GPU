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

        return;
    }
    void addReductionTest(){
        double *arr, *arr_dev[2];
        double sum;
        arr = (double*)malloc(100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[0], 100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[1], 100000 * sizeof(double));
        
        for(int i = 0; i < 100000; i++){
            arr[i] = 1.;
        }
        int flip = 0;
        cudaMemcpy(arr_dev[0], arr, 100000 * sizeof(double), cudaMemcpyHostToDevice);
        int remain;
        for(int len = 100000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&sum, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(99999.5 < sum && sum < 100000.5);

        for(int i = 0; i < 100000; i++){
            arr[i] = (double)i;
        }
        flip = 0;
        cudaMemcpy(arr_dev[0], arr, 100000 * sizeof(double), cudaMemcpyHostToDevice);
        for(int len = 100000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&sum, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(4.9999e9 < sum && sum < 5.0001e9);

        cudaFree(arr_dev[0]);
        cudaFree(arr_dev[1]);
        free(arr);

        return;
    }
}