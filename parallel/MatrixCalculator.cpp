#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#include "MatrixCalculator.h"

MatrixCalculator::MatrixCalculator(int numThreads) {
    this->numThreads = numThreads;
    this->threads = new pthread_t[numThreads];
    this->res = NULL;
}

// MatrixCalculator::MatrixCalculator() {
//     int numThreads = 4;
//     this->numThreads = numThreads;
//     this->threads = new pthread_t[numThreads];
// }

/*void MatrixCalculator::matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double** result_vec, int tid, int num_threads, barrier_t barrier) {
    result_vec = matrixTimesMatrix(mat, num_rows, num_cols, vec, vec_size, 1, tid, num_threads, barrier);
}*/

void MatrixCalculator::vectorTimesScalar(double* vec, int vec_size, double scalar, double* result) {
    for (int i = 0; i < vec_size; i++) {
        result[i] = vec[i] * scalar;
    }
}

double** MatrixCalculator::matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2, int tid, int num_threads, barrier_t barrier) {
    if(tid == 0) {
        if (num_cols1 != num_rows2) {
            printf("Matrix times matrix dimension error\n");
            exit(1);
        }

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

    this->barrier_exec(&barrier, num_threads);

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

    this->barrier_exec(&barrier, num_threads);
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

double** MatrixCalculator::transposeMatrix(double** mat, int num_rows, int num_cols) {
    double** res = this->allocate_2D(num_cols, num_rows);

    int new_num_threads = this->numThreads;
    if(num_rows < this->numThreads) new_num_threads = num_rows;
    int partitionSize = num_rows / new_num_threads;
    for(int i = 0; i < new_num_threads; i++) {
        transpose_thread_args * threadData = (transpose_thread_args*)malloc(sizeof(transpose_thread_args));
        if(i == numThreads - 1) {
            threadData->partitionSize = num_rows - (i*partitionSize);
            threadData->partitionStart = num_rows - threadData->partitionSize;
        } else {
            threadData->partitionSize = partitionSize;
            threadData->partitionStart = i * partitionSize;
        }
       
        threadData->tid = i;
        threadData->mat = mat;
        threadData->num_cols = num_cols;
        threadData->res = res;

        pthread_create(&(this->threads[i]), NULL, pTransposeMat, (void *)threadData);
    }

    for (int i = 0; i < new_num_threads; i++)  
        pthread_join(this->threads[i], NULL);     

    return res;
}

void MatrixCalculator::hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat) {
    int new_num_threads = this->numThreads;
    if(num_rows < this->numThreads) new_num_threads = num_rows;
    int partitionSize = num_rows / new_num_threads;
    for(int i = 0; i < new_num_threads; i++) {
        hadamard_thread_args * threadData = (hadamard_thread_args*)malloc(sizeof(hadamard_thread_args));
        if(i == numThreads - 1) {
            threadData->partitionSize = num_rows - (i*partitionSize);
            threadData->partitionStart = num_rows - threadData->partitionSize;
        } else {
            threadData->partitionSize = partitionSize;
            threadData->partitionStart = i * partitionSize;
        }
       
        threadData->tid = i;
        threadData->mat1 = mat1;
        threadData->mat2 = mat2;
        threadData->num_cols = num_cols;
        threadData->res = result_mat;

        pthread_create(&(this->threads[i]), NULL, pHadamardProd, (void *)threadData);
    }

    for (int i = 0; i < new_num_threads; i++)  
        pthread_join(this->threads[i], NULL);     
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

/**************** PARALLEL FUNCTIONS *******************************/
void MatrixCalculator::barrier_init(barrier_t *b) {
  b->count = 0;
  pthread_mutex_init(&(b->countLock), NULL);
  pthread_cond_init(&(b->okToProceed), NULL);
}

void MatrixCalculator::barrier_exec(barrier_t *b, int numThreads) {
  pthread_mutex_lock(&(b->countLock));
  b->count++;
  if(b->count == numThreads) {
    b->count = 0;
    pthread_cond_broadcast(&(b->okToProceed));
  } else {
    while(pthread_cond_wait(&(b->okToProceed), &(b->countLock)) != 0);
  }
  pthread_mutex_unlock(&(b->countLock));
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