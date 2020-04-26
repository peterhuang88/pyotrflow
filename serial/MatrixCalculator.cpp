#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "MatrixCalculator.h"

MatrixCalculator::MatrixCalculator() {
    // empty constructor...for now
}

void MatrixCalculator::matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double* result_vec) {
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
        for (int j = 0; j < num_cols; j++) {
            sum += mat[i][j] * vec[j][0];
        }
        result_vec[i] = sum;
    }
}

void MatrixCalculator::vectorTimesScalar(double* vec, int vec_size, double scalar, double* result) {
    for (int i = 0; i < vec_size; i++) {
        result[i] = vec[i] * scalar;
    }
}

double** MatrixCalculator::matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2) {
    // double** res = new double*[num_rows1]; 
    // for (int i = 0; i < num_rows1; i++) {
    //     res[i] = new double[num_cols2];
    // }
    if (num_cols1 != num_rows2) {
        printf("Matrix times matrix dimension error\n");
        exit(1);
    }

    double** res = this->allocate_2D(num_rows1, num_cols2);

    for (int i = 0; i < num_rows1; i++) {
        for (int j = 0; j < num_cols2; j++) {
            double sum = 0;
            for (int k = 0; k < num_cols1; k++) {
                sum += mat1[i][k] * mat2[k][j];
            }
            res[i][j] = sum;
        }
    }

    return res;

    // printf("Mat1: %d rows %d columns \nMat2: %d rows %d columns\n", num_rows1, num_cols1, num_rows2, num_cols2);

    // return NULL;
}

double** MatrixCalculator::transposeMatrix(double** mat, int num_rows, int num_cols) {
    // double** res = new double*[num_cols]; 
    // for (int i = 0; i < num_cols; i++) {
    //     res[i] = new double[num_rows];
    // }
    double** res = this->allocate_2D(num_cols, num_rows);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            res[j][i] = mat[i][j];
        }
    }

    return res;
}

void MatrixCalculator::hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            result_mat[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
}

double** MatrixCalculator::allocate_2D(int rows, int cols) {
    int i;             /* Loop variable              */
    double **pointers; /* The pointers for each row  */
    double *array;     /* The actually array of ints */

    /* Allocate memory for the rows X cols array */
    array = (double *) malloc(rows * cols * sizeof(double));

    /* Allocate the array of pointers, one per row */
    pointers = (double**)malloc(rows * sizeof(double *)); 

    /* Point each pointer at its corresponding row */
    for (i = 0; i < rows; i++) {
        pointers[i] = array + (cols * i);
    }

    return pointers;
}

void MatrixCalculator::free_2D(double** arr) {
    free(*arr);
    free(arr);
}

/*
int main(int argc, char** argv) {
    // double mat1[4][3] = {{1,5,9},{2,6,10},{3,7,11},{4,8,12}};
    // double mat2[3][1] = {{1},{2},{3}};
    double** mat1 = new double*[4];
    for (int i = 0; i < 4; i++) {
        mat1[i] = new double[3];
    }

    double** mat2 = new double*[3];
    for (int i = 0; i < 3; i++) {
        mat2[i] = new double[1];
    }
    
    mat1[0][0] = 1;
    mat1[0][1] = 2;
    mat1[0][2] = 3;
    mat1[1][0] = 4;
    mat1[1][1] = 5;
    mat1[1][2] = 6;
    mat1[2][0] = 7;
    mat1[2][1] = 8;
    mat1[2][2] = 9;
    mat1[3][0] = 10;
    mat1[3][1] = 11;
    mat1[3][2] = 12;

    mat2[0][0] = 1;
    mat2[1][0] = 2;
    mat2[2][0] = 3;

    MatrixCalculator mc;

    // double** temp = mc.matrixTimesMatrix(mat1, 4, 3, mat2, 3, 1);
    double** temp = mc.transposeMatrix(mat1, 4, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%lf ", temp[i][j]);
        }
        printf("\n");
    }

} */