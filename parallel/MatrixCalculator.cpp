#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#include "MatrixCalculator.h"

MatrixCalculator::MatrixCalculator() {

    this->res = NULL;
}

// MatrixCalculator::MatrixCalculator() {
//     int numThreads = 4;
//     this->numThreads = numThreads;
//     this->threads = new pthread_t[numThreads];
// }

/*void MatrixCalculator::matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double** result_vec, int tid, int num_threads, Barrier* barrier) {
    result_vec = matrixTimesMatrix(mat, num_rows, num_cols, vec, vec_size, 1, tid, num_threads, barrier);
}*/

void MatrixCalculator::vectorTimesScalar(double* vec, int vec_size, double scalar, double* result) {
    for (int i = 0; i < vec_size; i++) {
        result[i] = vec[i] * scalar;
    }
}

double** MatrixCalculator::matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2, int tid, int num_threads, Barrier* barrier) {
    if(tid == 0) {
        if (num_cols1 != num_rows2) {
            printf("Matrix times matrix dimension error\n");
            exit(1);
        }
        
        if(this->res != NULL) this->free_2D(this->res);
        this->res = this->allocate_2D(num_rows1, num_cols2);
    }

    int new_num_threads = num_threads;
    if(num_rows1 < num_threads) new_num_threads = num_rows1;
    int partitionSize = num_rows1 / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = num_rows1 - (tid*partitionSize);
        partitionStart = num_rows1 - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    barrier->barrier_exec(num_threads);

    if(tid < new_num_threads) {
        for (int i = partitionStart; i < partitionStart + partitionSize; i++)  {
            for (int j = 0; j < num_cols2; j++)  {
                double sum = 0;
                for (int k = 0; k < num_cols1; k++)  {
                    sum += mat1[i][k] * mat2[k][j]; 
                }
                this->res[i][j] = sum;
            }
        }   
    }

    barrier->barrier_exec(num_threads);
    return this->res;

    /*for (int i = 0; i < num_rows1; i++) {
        for (int j = 0; j < num_cols2; j++) {
            double sum = 0;
            for (int k = 0; k < num_cols1; k++) {
                sum += mat1[i][k] * mat2[k][j];
            }
            res[i][j] = sum;
        }
    }
    return res;*/
}

double** MatrixCalculator::transposeMatrix(double** mat, int num_rows, int num_cols, int tid, int num_threads, Barrier* barrier) {
    if(tid == 0) {
        if(this->res != NULL) this->free_2D(this->res);
        this->res = this->allocate_2D(num_cols, num_rows);
    }
    
    int new_num_threads = num_threads;
    if(num_rows < num_threads) new_num_threads = num_rows;
    int partitionSize = num_rows / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = num_rows - (tid*partitionSize);
        partitionStart = num_rows - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    barrier->barrier_exec(num_threads);  

    if(tid < new_num_threads) {
        for (int i = partitionStart; i < partitionStart + partitionSize; i++) {
            for (int j = 0; j < num_cols; j++) {
                this->res[j][i] = mat[i][j];
            }
        }
    }
        
    barrier->barrier_exec(num_threads);
    return this->res;
}

double** MatrixCalculator::hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, int tid, int num_threads, Barrier* barrier) {
    
    if(tid == 0) {
        if(this->res != NULL) this->free_2D(this->res);
        this->res = this->allocate_2D(num_rows, num_cols);
    }

    int new_num_threads = num_threads;
    if(num_rows < num_threads) new_num_threads = num_rows;
    int partitionSize = num_rows / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = num_rows - (tid*partitionSize);
        partitionStart = num_rows - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    barrier->barrier_exec(num_threads);
    if(tid < new_num_threads) {
        for (int i = partitionStart; i < partitionStart + partitionSize; i++) {
            for (int j = 0; j < num_cols; j++) {
                this->res[i][j] = mat1[i][j] * mat2[i][j];
            }
        }
    } 

    barrier->barrier_exec(num_threads);
    return this->res;
}

void MatrixCalculator::hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat, int tid, int num_threads, Barrier* barrier) {
    // Causing segfaults, currently using overloaded function
    int new_num_threads = num_threads;
    if(num_rows < num_threads) new_num_threads = num_rows;
    int partitionSize = num_rows / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = num_rows - (tid*partitionSize);
        partitionStart = num_rows - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    for (int i = partitionStart; i < partitionStart + partitionSize; i++) {
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