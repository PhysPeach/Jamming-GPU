#include <iostream>
#include "../hpp/MT.hpp"

#include "../testhpp/conf_test.hpp"
#include "../testhpp/particles_test.hpp"
#include "../testhpp/cells_test.hpp"
#include "../testhpp/jamming_test.hpp"

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

    //particles_test
    PhysPeach::createParticlesTest();
    PhysPeach::squeezePositionsTest();
    PhysPeach::powerParticlesTest();
    PhysPeach::convergedFireTest();
    PhysPeach::updateMemTest();
    PhysPeach::modifyVelocitiesTest();
    PhysPeach::UandPTest();
    PhysPeach::updateForcesTest();
    PhysPeach::updateParticlesTest();

    //cells_test
    PhysPeach::createCellsTest();
    PhysPeach::increaseNcTest();
    PhysPeach::updateCellsTest(); //memory warning: do it in small Np

    //lists_test
    PhysPeach::createListsTest();
    PhysPeach::increaseNlTest();
    PhysPeach::updateListsTest(); //memory warning: do it in small Np

    //jamming_test
    PhysPeach::createJammingTest();
    PhysPeach::fireJammingTest();
    //PhysPeach::addDphiTest();

    std::cout << "---finished---" << std::endl;
    return 0;
}