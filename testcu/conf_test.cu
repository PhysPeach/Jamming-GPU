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

    void multipliedTest(){

        int NB;

        double *arr, *arr_dev;
        arr = (double*)malloc(100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev, 100000 * sizeof(double));
        for(int i = 0; i < 100000; i++){
            arr[i] = 1.;
        }
        cudaMemcpy(arr_dev, arr, 100000 * sizeof(double), cudaMemcpyHostToDevice);

        NB = (100000 + NT - 1)/NT;
        multiplied<<<NB, NT>>>(arr_dev, 0.99, 100000);
        cudaMemcpy(arr, arr_dev, 100000 * sizeof(double), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 100000; i++){
            assert(0.989 < arr[i] && arr[i] < 0.991);
        }

        cudaFree(arr_dev);
        free(arr);

        return;
    }

    void glo_innerProductTest(){
        double *arr, *arr_dev[3];
        arr = (double*)malloc(100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[0], 100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[1], 100000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[2], 100000 * sizeof(double));

        for(int i = 0; i < 100000; i++){
            arr[i] = (double)(2*i);
        }
        cudaMemcpy(arr_dev[1], arr, 100000 * sizeof(double), cudaMemcpyHostToDevice);
        for(int i = 0; i < 100000; i++){
            arr[i] = (double)i;
        }
        cudaMemcpy(arr_dev[2], arr, 100000 * sizeof(double), cudaMemcpyHostToDevice);

        int NB = (100000 + NT - 1)/NT;
        glo_innerProduct<<<NB, NT>>>(arr_dev[0], arr_dev[1], arr_dev[2], 100000);
        cudaMemcpy(arr, arr_dev[0], 100000 * sizeof(double), cudaMemcpyDeviceToHost);

        for(int i = 0; i < 100000; i++){
            assert((double)(2. * i) * (double)i - 0.1 < arr[i] && arr[i] < (double)(2. * i) * (double)i + 0.1);
        }

        cudaFree(arr_dev[0]);
        cudaFree(arr_dev[1]);
        cudaFree(arr_dev[2]);
        free(arr);
    }
}