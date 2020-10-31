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
}