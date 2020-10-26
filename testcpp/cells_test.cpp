#include "../testhpp/cells_test.hpp"

namespace PhysPeach{
    //cellsTest
    void createCellsTest(){

        Cells cells;
        createCells(&cells, 3.);
        assert(cells.numOfCellsPerSide == 3);
        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(3, D)));
        deleteCells(&cells);

        createCells(&cells, 40.);
        assert(cells.numOfCellsPerSide == 12);
        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(12, D)));
        deleteCells(&cells);

        return;
    }

    void increaseNcTest(){
        Cells cells;
        createCells(&cells, 40.);

        assert(cells.Nc == (int)(3. * (double)Np/ (double)powInt(12, D)));
        increaseNc(&cells);
        assert(cells.Nc == (int)(1.4 * (int)(3. * (double)Np/ (double)powInt(12, D))));
        deleteCells(&cells);

        return;
    }

    void updateCellsTest(){

        Cells cells;

        if(Np <= 100){
            //test it in small Np
            double *x;
            x = (double*)malloc(D*Np*sizeof(double));
            for(int par1 = 0; par1 < D*Np; par1++){
                x[par1] = 1.;
            }
            createCells(&cells, 10.);
            updateCells(&cells, 10., x);
            assert(cells.Nc == Np+1);
            deleteCells(&cells);
            free(x);
        }

        Particles p;
        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);
        createCells(&cells, L);
        updateCells(&cells, L, p.x);
        deleteCells(&cells);
        deleteParticles(&p);

        return;
    }

    //listsTest
    void createListsTest(){

        Cells cells;
        Lists lists;
        createCells(&cells, 3.);
        createLists(&lists, &cells);
        assert(lists.Nl == (int)(3.2 * (int)(3. * (double)Np/ (double)powInt(3, D))));
        deleteLists(&lists);
        deleteCells(&cells);

        createCells(&cells, 40.);
        createLists(&lists, &cells);
        assert(lists.Nl == (int)(3.2 * (int)(3. * (double)Np/ (double)powInt(12, D))));
        deleteLists(&lists);
        deleteCells(&cells);

        return;
    }

    void increaseNlTest(){
        Cells cells;
        Lists lists;

        createCells(&cells, 40.);
        createLists(&lists, &cells);

        assert(lists.Nl == (int)(3.2 * (int)(3. * (double)Np/ (double)powInt(12, D))));
        increaseNl(&lists);
        assert(lists.Nl == (int)(1.4 * (int)(3.2 * (int)(3. * (double)Np/ (double)powInt(12, D)))));

        deleteLists(&lists);
        deleteCells(&cells);

        return;
    }

    void updateListsTest(){

        Cells cells;
        Lists lists;

        if(Np <= 100){
            //test it in small Np
            double *x;
            x = (double*)malloc(D*Np*sizeof(double));
            for(int par1 = 0; par1 < D*Np; par1++){
                x[par1] = 1.;
            }
            createCells(&cells, 10.);
            createLists(&lists, &cells);

            updateCells(&cells, 10., x);
            updateLists(&lists, &cells, 10., x);
            assert(lists.Nl == Np);
            assert(lists.list[0] == Np - 1);
            for(int i = 1; i <= lists.list[0]; i++){
                assert(lists.list[i] == i);
            }
            assert(lists.list[lists.Nl] == Np -2);
            for(int i = 1; i < lists.list[lists.Nl]; i++){
                assert(lists.list[lists.Nl+i] == i+1);
            }
            deleteLists(&lists);
            deleteCells(&cells);
            free(x);
        }

        Particles p;
        createParticles(&p);
        double L = pow(p.packing/Phi_init, 1./(double)D);
        createCells(&cells, L);
        createLists(&lists, &cells);
        updateCells(&cells, L, p.x);
        updateLists(&lists, &cells, L, p.x);
        deleteLists(&lists);
        deleteCells(&cells);
        deleteParticles(&p);

        return;
    }
}