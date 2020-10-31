#include "../testcuh/jamming_test.cuh"

namespace PhysPeach{
    void createJammingTest(){
        Jamming jam;
        createJamming(&jam);
        //std::cout << jam.lists.Nl << std::endl;
        deleteJamming(&jam);
        return;
    }

    void fireJammingTest(){
        Jamming jam;
        createJamming(&jam);
        fireJamming(&jam);

        double *arr;
        arr = (double*)malloc(D*Np*sizeof(double));
        cudaMemcpy(arr, jam.p.f_dev, D*Np*sizeof(double), cudamemcpyDeviceToHost);

        double fsum = 0.;
        double f2;
        for(int par1 = 0; par1 < Np; par1++){
            f2 = 0.;
            for(int d = 0; d < D; d++){
                f2 += arr[par1+d*Np] * arr[par1+d*Np];
            }
            fsum += sqrt(f2);
        }
        assert(fsum < 1.0e-13 * Np);
        free(arr);
        deleteJamming(&jam);
        return;
    }
}