#ifndef CONF_CUH
#define CONF_CUH

#include <math.h>

//gpu
const int NT = 1024;

const double pi = 3.141592653589793;

const int D = 2;
extern int ID;
extern int Np;
extern double Phi_init;
extern double Dphi;
const double a0 = 1.;
const double a_min = 0.7253;
const double a_max = 1.6095;
const double A = 1.3203;

const double dt_init = 0.005;
const double dt_max = 0.05;
const double alpha_init = 0.1;

namespace PhysPeach{
    template<typename T>
    T powInt(T, int);

    template<typename T>
    void setZero(T*, int);

    //global
    __global__ void fillSameNum_int(int*, int, int);
    __global__ void addReduction(double*, double*, int);
    __global__ void maxReduction(double*, double*, int);
    __global__ void multiplied(double*, double , int);
    __global__ void glo_innerProduct(double*, double*, double*, int);
    __global__ void absolute(double* , double* , int);
}
#endif