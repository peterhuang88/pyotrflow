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

void MatrixCalculator::matrixTimesVector(double** mat, int num_rows, int num_cols, double* vec, int vec_size, double* result_vec) {
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
        for (int j = 0; j < num_cols; j++) {
            sum += mat[i][j] * vec[j];
        }
        result_vec[i] = sum;
    }
}