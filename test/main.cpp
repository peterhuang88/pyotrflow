#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>

#include "MatrixCalculator.h"

int main(int argc, char** argv) {
    int num_rows1 = 4;
    int num_cols1 = 4;
    int num_rows2 = 4;
    int num_cols2 = 4;
    double** mat1 = new double*[num_rows1];
    for (int i = 0; i < num_rows1; i++) {
        mat1[i] = new double[num_cols1];
    }

    double** mat2 = new double*[num_rows2];
    for (int i = 0; i < num_rows2; i++) {
        mat2[i] = new double[num_cols2];
    }

    double** res = new double*[num_rows2];
    for (int i = 0; i < num_rows2; i++) {
        res[i] = new double[num_cols2];
    }

    // Generating random values in mat1 and mat2 
    for (int i = 0; i < num_rows1; i++) { 
        for (int j = 0; j < num_cols1; j++) { 
            mat1[i][j] = 2; 
        } 
    } 

    for (int i = 0; i < num_rows2; i++) { 
        for (int j = 0; j < num_cols2; j++) { 
            mat2[i][j] = 2; 
        } 
    } 

    int numThreads = 16;
    MatrixCalculator mc(numThreads);

    //double** temp = mc.matrixTimesMatrix(mat1, num_rows1, num_cols1, mat2, num_rows2, num_cols2);
    mc.hadamardProduct(mat1, mat2, num_rows1, num_cols1, res);
    for (int i = 0; i < num_rows1; i++) {
        for (int j = 0; j < num_cols1; j++) {
            printf("%lf ", res[i][j]);
        }
        printf("\n");
    }

}