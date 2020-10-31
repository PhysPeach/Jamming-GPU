#include "../cuh/jamming.cuh"

namespace PhysPeach{
    double L(Jamming* jam){
        return pow(jam->p.packing/jam->phi, 1./(double)D);
    }

    void createJamming(Jamming* jam){
        jam->phi = Phi_init;
        createParticles(&jam->p);
        createCells(&jam->cells, L(jam));
        createLists(&jam->lists, &jam->cells);
        updateCellList(&jam->cells, &jam->lists, L(jam), jam->p.x_dev);
        return;
    }

    void loadSwapMC(Jamming* jam){
        jam->phi = Phi_init;
        
        std::ostringstream posName;
        posName << "../../swapmc/pos/";
        posName << "pos_N" << Np << "_Phi" << Phi_init << "_id" << ID << ".data";
        std::ifstream file;
        file.open(posName.str().c_str());
        createParticles(&jam->p, &file);
        file.close();
        createCells(&jam->cells, L(jam));
        createLists(&jam->lists, &jam->cells);
        updateCellList(&jam->cells, &jam->lists, L(jam), jam->p.x_dev);
        return;
    }

    void loadJamming(Jamming* jam){

        std::ifstream file;

        std::ostringstream jammingName;
        jammingName << "../jammingpoint/jam_N" << Np << "_Phi" << Phi_init << "_id" << ID <<".data";
        file.open(jammingName.str().c_str());
        file >> jam->phi;
        file.close();
        
        std::ostringstream posName;
        posName << "../pos/pos_N" << Np << "_Phi" << Phi_init << "_id" << ID << ".data";
        file.open(posName.str().c_str());
        createParticles(&jam->p, &file);
        file.close();

        createCells(&jam->cells, L(jam));
        createLists(&jam->lists, &jam->cells);
        updateCellList(&jam->cells, &jam->lists, L(jam), jam->p.x_dev);
        return;
    }

    void deleteJamming(Jamming* jam){
        deleteParticles(&jam->p);
        deleteLists(&jam->lists);
        deleteCells(&jam->cells);
        return;
    }

    int fireJamming(Jamming* jam){
        int loop = 0;
        double dt = dt_init;
        double alpha = alpha_init;
        double power;

        bool converged = false;
        int cp = 0;
        fillSameNum_double<<<(D*Np + NT - 1)/NT, NT>>>(jam->p.v_dev, 0., D*Np);
        while(!converged){
            loop++;
            updateParticles(&jam->p, L(jam), dt, &jam->lists);
            checkUpdateCellList(&jam->cells, &jam->lists, L(jam), jam->p.x_dev, jam->p.v_dev);
            converged = convergedFire(&jam->p);
            power = powerParticles(&jam->p);
            modifyVelocities<<<(Np + NT - 1)/NT, NT>>>(jam->p.v_dev, jam->p.f_dev, alpha, Np);
            if(power < 0){
                fillSameNum_double<<<(D*Np + NT - 1)/NT, NT>>>(jam->p.v_dev, 0., D*Np);
                alpha = alpha_init;
                dt *= 0.5;
                cp = 0;
            }else{
                cp++;
                if(cp > 5){
                    dt *= 1.1;
                    if(dt > dt_max){
                        dt = dt_max;
                    }
                    alpha *= 0.99;
                    cp = 0;
                }
            }
            if(loop == 1000000){
                std::cout << "dt: " << dt << std::endl;
            }
        }
        return loop;
    }
}