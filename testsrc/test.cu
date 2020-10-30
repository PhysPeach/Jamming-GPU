#include <iostream>
#include "../cuh/MT.cuh"

#include "../testcuh/conf_test.cuh"
#include "../testcuh/particles_test.cuh"
#include "../testcuh/cells_test.cuh"
#include "../testcuh/jamming_test.cuh"

int ID;
int Np;
double Phi_init;
double Dphi;

int main(){
    ID = 0;
    Np = 1024;
    Phi_init = 0.8;
    Dphi = 0.05;

    std::cout << "--start test--" << std::endl;
    init_genrand(1);

    //conf_test
    PhysPeach::powIntTest();
    PhysPeach::setZeroTest();
    PhysPeach::addReductionTest();
    PhysPeach::maxReductionTest();
    PhysPeach::multipliedTest();
    PhysPeach::glo_innerProductTest();
    PhysPeach::absoluteTest();

    //particles_test
    PhysPeach::createParticlesTest();
    PhysPeach::powerParticlesTest();
    PhysPeach::convergedFireTest();
    PhysPeach::modifyVelocitiesTest();
    //PhysPeach::updateMemTest();
    //PhysPeach::UandPTest();
    //PhysPeach::updateForcesTest();
    //PhysPeach::updateParticlesTest();

    //cells_test
    PhysPeach::createCellsTest();
    PhysPeach::increaseNcTest();
    PhysPeach::updateCellsTest();

    //lists_test
    PhysPeach::createListsTest();
    PhysPeach::increaseNlTest();
    PhysPeach::updateListsTest(); //memory warning: do it in small Np

    //cellList_test
    PhysPeach::setUpdateFreqTest();

    //jamming_test
    //PhysPeach::createJammingTest();
    //PhysPeach::fireJammingTest();
    //PhysPeach::addDphiTest();

    std::cout << "---finished---" << std::endl;
    return 0;
}