#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#include "Barrier.h"

#ifndef MATRIXCALCULATOR_H
#define MATRIXCALCULATOR_H

struct transpose_thread_args {
    int tid;
    double ** mat;
    int num_cols;
    int partitionSize;
    int partitionStart;
    double **res;
};

struct hadamard_thread_args {
    int tid;
    double ** mat1;
    double ** mat2;
    int num_cols;
    int partitionSize;
    int partitionStart;
    double **res;
};

class MatrixCalculator {
    public:
        MatrixCalculator(int numThreads);

        // actually useful functions
        void matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double** result_vec, int tid, int num_threads, Barrier* barrier);
        void vectorTimesScalar(double* vec, int vec_size, double scalar, double* result);
        double** matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2, int tid, int num_threads, Barrier* barrier);
        double** transposeMatrix(double** mat, int num_rows, int num_cols);
        void hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat); // element wise product of 2 matrices

        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // random helper functions

        int numThreads;
        pthread_t * threads;
        double** res;
        
    private:
        static void* pTransposeMat(void * thread_args) {
            transpose_thread_args * args = (transpose_thread_args*) thread_args;
            for (int i = args->partitionStart; i < args->partitionStart + args->partitionSize; i++) {
                for (int j = 0; j < args->num_cols; j++) {
                    args->res[j][i] = args->mat[i][j];
                }
            }
        }  

        static void* pHadamardProd(void * thread_args) {
            hadamard_thread_args * args = (hadamard_thread_args*) thread_args;
            for (int i = args->partitionStart; i < args->partitionStart + args->partitionSize; i++) {
                for (int j = 0; j < args->num_cols; j++) {
                    args->res[i][j] = args->mat1[i][j] * args->mat2[i][j];
                }
            }
        }  
};

#endif