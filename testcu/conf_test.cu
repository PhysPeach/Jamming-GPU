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
        arr = (double*)malloc(10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[0], 10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[1], 10000 * sizeof(double));
        
        for(int i = 0; i < 10000; i++){
            arr[i] = 1.;
        }
        int flip = 0;
        cudaMemcpy(arr_dev[0], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);
        int remain;
        for(int len = 10000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&sum, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(9999.5 < sum && sum < 10000.5);

        for(int i = 0; i < 10000; i++){
            arr[i] = (double)i;
        }
        flip = 0;
        cudaMemcpy(arr_dev[0], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);
        for(int len = 10000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            addReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&sum, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(4.99e7 < sum && sum < 5.01e7);

        cudaFree(arr_dev[0]);
        cudaFree(arr_dev[1]);
        free(arr);

        return;
    }

    void maxReductionTest(){
        double *arr, *arr_dev[2];
        double max;
        arr = (double*)malloc(10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[0], 10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[1], 10000 * sizeof(double));
        
        arr[0] = 1.;
        for(int i = 1; i < 10000; i++){
            arr[i] = 0.;
        }
        int flip = 0;
        cudaMemcpy(arr_dev[0], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);
        int remain;
        for(int len = 10000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            maxReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&max, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(0.999 < max && max < 1.001);

        for(int i = 0; i < 10000; i++){
            arr[i] = (double)i;
        }
        flip = 0;
        cudaMemcpy(arr_dev[0], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);
        for(int len = 10000; len > 1; len = remain){
            remain = (len+NT-1)/NT;
            flip = !flip;
            maxReduction<<<remain,NT>>>(arr_dev[flip], arr_dev[!flip], len);
        }
        cudaMemcpy(&max, arr_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
        assert(9998.999 < max && max < 9999.001);

        cudaFree(arr_dev[0]);
        cudaFree(arr_dev[1]);
        free(arr);

        return;
    }

    void multipliedTest(){

        int NB;

        double *arr, *arr_dev;
        arr = (double*)malloc(10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev, 10000 * sizeof(double));
        for(int i = 0; i < 10000; i++){
            arr[i] = 1.;
        }
        cudaMemcpy(arr_dev, arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);

        NB = (10000 + NT - 1)/NT;
        multiplied<<<NB, NT>>>(arr_dev, 0.99, 10000);
        cudaMemcpy(arr, arr_dev, 10000 * sizeof(double), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 10000; i++){
            assert(0.989 < arr[i] && arr[i] < 0.991);
        }

        cudaFree(arr_dev);
        free(arr);

        return;
    }

    void glo_innerProductTest(){
        double *arr, *arr_dev[3];
        arr = (double*)malloc(10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[0], 10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[1], 10000 * sizeof(double));
        cudaMalloc((void**)&arr_dev[2], 10000 * sizeof(double));

        for(int i = 0; i < 10000; i++){
            arr[i] = (double)(2*i);
        }
        cudaMemcpy(arr_dev[1], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);
        for(int i = 0; i < 10000; i++){
            arr[i] = (double)i;
        }
        cudaMemcpy(arr_dev[2], arr, 10000 * sizeof(double), cudaMemcpyHostToDevice);

        int NB = (10000 + NT - 1)/NT;
        glo_innerProduct<<<NB, NT>>>(arr_dev[0], arr_dev[1], arr_dev[2], 10000);
        cudaMemcpy(arr, arr_dev[0], 10000 * sizeof(double), cudaMemcpyDeviceToHost);

        for(int i = 0; i < 10000; i++){
            assert((double)(2. * i) * (double)i - 0.1 < arr[i] && arr[i] < (double)(2. * i) * (double)i + 0.1);
        }

        cudaFree(arr_dev[0]);
        cudaFree(arr_dev[1]);
        cudaFree(arr_dev[2]);
        free(arr);
    }

    void absoluteTest(){

        double *arr, *arr_dev, *arrabs_dev;
        arr = (double*)malloc(D * 5000 * sizeof(double));
        cudaMalloc((void**)&arr_dev, D * 5000 * sizeof(double));
        cudaMalloc((void**)&arrabs_dev, 5000 * sizeof(double));
        for(int i = 0; i < D * 5000; i++){
            arr[i] = (double)i;
        }
        cudaMemcpy(arr_dev, arr, D * 5000 * sizeof(double), cudaMemcpyHostToDevice);
        absolute<<<(D * 5000 + NT - 1)/NT, NT>>>(arrabs_dev, arr_dev, 5000);
        cudaMemcpy(arr, arrabs_dev, 5000 * sizeof(double), cudaMemcpyDeviceToHost);
        for(int i = 0; i < 5000; i++){
            if(D == 2){
                assert(sqrt((double)i*(double)i+((double)i + 5000.)*((double)i + 5000.)) - 0.1 < arr[i] && arr[i] < sqrt((double)i*(double)i+((double)i + 5000.)*((double)i + 5000.)) + 0.1);
            }else if(D == 3){
                assert(sqrt((double)i*(double)i+((double)i + 5000.)*((double)i + 5000.)+((double)i + 10000.)*((double)i + 10000.)) - 0.1 < arr[i] && arr[i] < sqrt((double)i*(double)i+((double)i + 5000.)*((double)i + 5000.)+((double)i + 10000.)*((double)i + 10000.)) + 0.1);
            }
        }

        cudaFree(arrabs_dev);
        cudaFree(arr_dev);
        free(arr);

        return;
    }
}