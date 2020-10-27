#include "../cuh/conf.cuh"

namespace PhysPeach{
    template<typename T>
    T powInt(T a, int x){
        T result = a/a;
        if (x >= 0){
            for (int i = 0; i < x; i++){
                result *= a;
            }
        }else{
            for (int i = 0; i < -x; i++){
                result /= a;
            }
        }
        return result;
    }
    template int powInt<int>(int, int);
    template double powInt<double>(double, int);

    template<typename T>
    void setZero(T *arr, int Narr){
        for(int arr1 = 0; arr1 < Narr; arr1++){
            arr[arr1] -= arr[arr1];
        }
        return;
    }
    template void setZero<int>(int*, int);
    template void setZero<double>(double*, int);

    //device
    __global__ void addReduction(double *out, double *in, int len){
        int i_block = blockIdx.x;
        int i_local = threadIdx.x;
        int i_global = i_block * blockDim.x + i_local;
    
        __shared__ double f[NT];
    
        if(i_global < len){
            f[i_local] = in[i_global];
        }
        __syncthreads();

        int remain, reduce;
        for(int j = NT; j > 1; j = remain){
            reduce = j >> 1;
            remain = j - reduce;
            if((i_local < reduce) && (i_global + remain < len)){
                f[i_local] += f[i_local+remain];
            }
            __syncthreads();
        }
        if(i_local == 0){
            out[i_block] = f[0];
        }
    }

    __global__ void multiplied(double *arr_dev, double a, int len){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < len){
            arr_dev[i_global] *= a;
        }
    }

    __global__ void glo_innerProduct(double* out_dev, double* in1_dev, double* in2_dev, int len){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;
        if(i_global < len){
            out_dev[i_global] = in1_dev[i_global] * in2_dev[i_global];
        }
    }

    __global__ void absolute(double* abs_dev, double* vec, int len){
        int i_global = blockIdx.x * blockDim.x + threadIdx.x;

        if(i_global < len){
            abs_dev[i_global] = 0.;
            for(int d = 0; d < D; d++){
                abs_dev[i_global] += vec[d*len + i_global] * vec[d*len + i_global];
            }
            abs_dev[i_global] = sqrt(abs_dev[i_global]);
        }
    }
}