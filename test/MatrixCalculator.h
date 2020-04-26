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

#ifndef MATRIXCALCULATOR_H
#define MATRIXCALCULATOR_H

//double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2
struct matmat_thread_args { 
    int tid;
    double ** mat1; 
    int num_cols1;
    double ** mat2;
    int num_cols2;
    int partitionSize;
    int partitionStart;
    double ** res;
};

struct transpose_thread_args {
    int tid;
    double ** mat;
    int num_cols;
    int partitionSize;
    int partitionStart;
    double **res;
};

class MatrixCalculator {
    public:
        MatrixCalculator(int numThreads);

        // actually useful functions
        void matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double* result_vec);
        void vectorTimesScalar(double* vec, int vec_size, double scalar, double* result);
        double** matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2);
        double** transposeMatrix(double** mat, int num_rows, int num_cols);
        void hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat); // element wise product of 2 matrices

        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // random helper functions
        

        int numThreads;
        pthread_t * threads;
        
    private:
        static void* pMatTimesMat(void * thread_args) {
            matmat_thread_args * args = (matmat_thread_args*) thread_args;
            for (int i = args->partitionStart; i < args->partitionStart + args->partitionSize; i++)  {
                for (int j = 0; j < args->num_cols2; j++)  {
                    double sum = 0;
                    for (int k = 0; k < args->num_cols1; k++)  {
                        sum += args->mat1[i][k] * args->mat2[k][j]; 
                    }
                    args->res[i][j] = sum;
                }
            }    
        }

        static void* pTransposeMat(void * thread_args) {
            transpose_thread_args * args = (transpose_thread_args*) thread_args;
                for (int i = args->partitionStart; i < args->partitionStart + args->partitionSize; i++) {
                    for (int j = 0; j < args->num_cols; j++) {
                        args->res[j][i] = args->mat[i][j];
                    }
                }
        }  
};

#endif