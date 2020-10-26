#include "../hpp/conf.hpp"

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
}