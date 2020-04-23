#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#ifndef MATRIXCALCULATOR_H
#define MATRIXCALCULATOR_H

class MatrixCalculator {
    public:
        MatrixCalculator();

        // actually useful functions
        void matrixTimesVector(double** mat, int num_rows, int num_cols, double* vec, int vec_size, double* result_vec);

        // random helper functions
        

        // debug functions
        
        
};

#endif