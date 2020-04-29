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

class MatrixCalculator {
    public:
        MatrixCalculator();

        // actually useful functions
        void matrixTimesVector(double** mat, int num_rows, int num_cols, double** vec, int vec_size, double** result_vec, int tid, int num_threads, Barrier* barrier);
        void vectorTimesScalar(double* vec, int vec_size, double scalar, double* result);
        double** matrixTimesMatrix(double** mat1, int num_rows1, int num_cols1, double** mat2, int num_rows2, int num_cols2, int tid, int num_threads, Barrier* barrier);
        double** transposeMatrix(double** mat, int num_rows, int num_cols, int tid, int num_threads, Barrier* barrier);
        void hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, double** result_mat, int tid, int num_threads, Barrier* barrier); // element wise product of 2 matrices
        double** hadamardProduct(double** mat1, double** mat2, int num_rows, int num_cols, int tid, int num_threads, Barrier* barrier); // element wise product of 2 matrices

        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // random helper functions
        double** res;
};

#endif