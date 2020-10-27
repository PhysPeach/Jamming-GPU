#include "../cuh/cells.cuh"

namespace PhysPeach{
    //cells
    void createCells(Cells *cells, double L){
        cells->numOfCellsPerSide = (int)(L/(2. * a_max));
        if(cells->numOfCellsPerSide < 3){
            cells->numOfCellsPerSide = 3;
        }
        double buf = 3.;
        cells->Nc = (int)(buf * (double)Np/ (double)powInt(cells->numOfCellsPerSide, D));

        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        cells->cell = (int*)malloc(NoC*sizeof(int));
        setZero(cells->cell, NoC);
        return;
    }
    
    void deleteCells(Cells *cells){
        free(cells->cell);
        return;
    }
    
    void increaseNc(Cells *cells){
        cells->Nc = (int)(1.4 * cells->Nc);
        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        cells->cell = (int*)realloc(cells->cell, NoC*sizeof(int));
        setZero(cells->cell, NoC);
        return;
    }

    bool putParticlesIntoCells(Cells *cells, double L, double* x){
        int c1, c2, c3;
        double Lc = L/(double)cells->numOfCellsPerSide;
        double Lh = 0.5 * L;
        int NoC = powInt(cells->numOfCellsPerSide, D)*cells->Nc;
        setZero(cells->cell, NoC);
        for(int par1 = 0; par1 < Np; par1++){
            c1 = (x[par1] + Lh)/Lc;
            c2 = (x[par1+Np] + Lh)/Lc;
            if(c1 >= cells->numOfCellsPerSide){c1 -= cells->numOfCellsPerSide;}
            if(c1 < 0){c1 += cells->numOfCellsPerSide;}
            if(c2 >= cells->numOfCellsPerSide){c2 -= cells->numOfCellsPerSide;}
            if(c2 < 0){c2 += cells->numOfCellsPerSide;}
            if (D == 2){
                cells->cell[(c1*cells->numOfCellsPerSide+c2)*cells->Nc]++;
                if(cells->cell[(c1*cells->numOfCellsPerSide+c2)*cells->Nc] >= cells->Nc){
                    return false;
                }
                cells->cell[(c1*cells->numOfCellsPerSide+c2)*cells->Nc + cells->cell[(c1*cells->numOfCellsPerSide+c2)*cells->Nc]] = par1;
            }else{
                std::cout << "dimention err" << std::endl;
                    exit(1);
            }
        }
        return true;
    }

    void updateCells(Cells *cells, double L, double* x){
        bool success = false;
        success = putParticlesIntoCells(cells, L, x);
        while (!success){
            std::cout << "hello" << std::endl;
            increaseNc(cells);
            success = putParticlesIntoCells(cells, L, x);
        }

        return;
    }

    //lists
    void createLists(Lists *lists, Cells *cells){
        lists->Nl = (int)(3.2 * (double)cells->Nc);

        int NoL = lists->Nl * Np;
        lists->list = (int*)malloc(NoL*sizeof(int));
        setZero(lists->list, NoL);
        return;
    }
    
    void deleteLists(Lists *lists){
        free(lists->list);
        return;
    }
    
    void increaseNl(Lists *lists){
        lists->Nl = (int)(1.4 * lists->Nl);
        int NoL = lists->Nl * Np;
        lists->list = (int*)realloc(lists->list, NoL*sizeof(int));
        setZero(lists->list, NoL);
        return;
    }

    int fix(int i, int M) {
	    if (i < 0) i += M;
	    if (i >= M)i -= M;

	    return i;
    }

    bool putParticlesIntoLists(Lists *lists, Cells *cells, double L, double* x){
        int c1, c2, c3;
        int g1, g2, g3;
        int numOfParticlesInCell;
        double Lc = L/(double)cells->numOfCellsPerSide;
        double Lh = 0.5 * L;

        int NoL = lists->Nl * Np;
        setZero(lists->list, NoL);

        double x1[D];
        int par2;
        double dx[D], dr;

        for(int par1 = 0; par1< Np; par1++){
            for(int d = 0; d < D; d++){
                x1[d] = x[d*Np+par1];
            }
            c1 = (x1[0] + Lh)/Lc;
            c2 = (x1[1] + Lh)/Lc;
            if(c1 >= cells->numOfCellsPerSide){c1 -= cells->numOfCellsPerSide;}
            if(c1 < 0){c1 += cells->numOfCellsPerSide;}
            if(c2 >= cells->numOfCellsPerSide){c2 -= cells->numOfCellsPerSide;}
            if(c2 < 0){c2 += cells->numOfCellsPerSide;}

            for(int e1 = c1-1; e1 <= c1+1; e1++){
                g1 = fix(e1, cells->numOfCellsPerSide);
                for(int e2= c2-1; e2 <= c2+1; e2++){
                    g2 = fix(e2, cells->numOfCellsPerSide);
                    numOfParticlesInCell = cells->cell[(g1*cells->numOfCellsPerSide+g2)*cells->Nc];
                    for(int k = 1; k<= numOfParticlesInCell;k++){
                        par2 = cells->cell[(g1*cells->numOfCellsPerSide+g2)*cells->Nc + k];
                        if(par2 > par1){
                            dr = 0.;
                            for(int d = 0; d < D; d++){
                                dx[d] = x1[d] - x[d*Np+par2];
                                if(dx[d] < -Lh) dx[d] += L;
                                if(dx[d] > Lh) dx[d] -= L;
                                dr += dx[d] * dx[d];
                            }
                            if(dr < 4 * a_max * a_max){
                                lists->list[par1*lists->Nl]++;
                                if(lists->list[par1*lists->Nl] >= lists->Nl){
                                    return false;
                                }
                                lists->list[par1*lists->Nl + lists->list[par1*lists->Nl]] = par2;
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    void updateLists(Lists *lists, Cells *cells, double L, double* x){
        bool success = false;
        success = putParticlesIntoLists(lists, cells, L, x);
        while (!success){
            std::cout << "hi" << std::endl;
            increaseNl(lists);
            success = putParticlesIntoLists(lists, cells, L, x);
        }

        return;
    }

    void updateCellList(Cells *cells, Lists *lists, double L, double* x){
        updateCells(cells, L, x);
        updateLists(lists, cells, L, x);

        return;
    }
}