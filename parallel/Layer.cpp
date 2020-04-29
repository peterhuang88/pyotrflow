#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#include "Layer.h"

/**
* num_neurons = number of neurons in current layer
* num_input = number of inputs from previous layer e.g. number of neurons in previous layer
*/

Layer::Layer(int num_input, int num_neurons, int marker, std::string name, int num_threads) {
    this->num_neurons = num_neurons;
    this->num_input = num_input;
    this->name = name;
    this->marker = marker;

    // this->W = new double*[num_neurons];
    // for (int i = 0; i < num_neurons; i++) {
    //     this->W[i] = new double[num_input];
    // }
    this->W = this->allocate_2D(num_neurons, num_input);

    // this->Z = new double[num_neurons];
    this->Z = this->allocate_2D(num_neurons, 1);

    // this->A = new double*[num_neurons];
    // // TODO only works for fc layers
    // for (int i = 0; i < num_neurons; i++) {
    //     this->A[i] = new double[1];
    // }
    this->A = this->allocate_2D(num_neurons, 1);

    
    this->b = new double[num_neurons];
    this->num_threads = num_threads;
    this->sigmoid_deriv_ret = NULL;
    // TODO: THIS NEEDS TO CHANGE
    // this->gradients = new double[num_neurons];
    // this->initializeTestWeights();
}

Layer::~Layer() {
    // for (int i = 0; i < this->num_input; i++) {
    //     delete [] W[i];
    // }
    // delete [] W;
    free_2D(this->W);
    // delete [] this->Z;
    free_2D(this->Z);
    free_2D(this->A);
    delete [] this->b;
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Layer::backProp(double** W_next, int W_next_rows, int W_next_cols, double** dZ_next, int dZ_next_rows, int dZ_next_cols, double** A_prev, int A_prev_rows, int A_prev_cols, int tid, Barrier* barrier) {
    // calculate dZ
    double** W_next_transpose = mc.transposeMatrix(W_next, W_next_rows, W_next_cols, tid, this->num_threads, barrier);
     
    barrier->barrier_exec(this->num_threads);
    double** temp1 = mc.matrixTimesMatrix(W_next_transpose, W_next_cols, W_next_rows, dZ_next, dZ_next_rows, dZ_next_cols, tid, this->num_threads, barrier);
    printf("this->Z[0]: %p\n", this->Z[0]);
    double** deriv = this->sigmoid_derivative(this->Z[0], this->num_neurons, tid, barrier);
    //printf("sigmoid done! %d\n", tid);
    
    barrier->barrier_exec(this->num_threads);
   
    this->dZ = mc.hadamardProduct(temp1, deriv, this->num_neurons, 1, tid, this->num_threads, barrier);

}

void Layer::lastLayerBackProp(double Y, double** A_prev, int A_prev_rows, int A_prev_cols, int tid, Barrier* barrier) {
    this->dZ[0][0] = this->A[0][0] - Y;
    double** A_prev_transpose = mc.transposeMatrix(A_prev, A_prev_rows, A_prev_cols, tid, this->num_threads, barrier);
    //this->free_2D(this->dW);
    barrier->barrier_exec(this->num_threads);
    
    this->dW = mc.matrixTimesMatrix(this->dZ, this->dZ_rows, this->dZ_cols, A_prev_transpose, A_prev_cols, A_prev_rows, tid, this->num_threads, barrier);
    //this->free_2D(this->dB);
    this->dB = this->dZ;

    // deallocate A_prev_transpose because we made alloc'd a new array for it
    //this->free_2D(A_prev_transpose);
}

void Layer::forwardProp(double** input, int tid, Barrier* barrier) {
    // perform wTx 
    
    this->Z = mc.matrixTimesMatrix(this->W, num_neurons, num_input, input, num_input, 1, tid, this->num_threads, barrier);

    barrier->barrier_exec(this->num_threads);
    // mc.matrixTimesVector(this->W, num_neurons, num_input, input, num_neurons, this->Z);
    
    // add bias to each z
    // TODO: potentially parallelize

    int new_num_threads = this->num_threads;
    if(this->num_neurons < this->num_threads) new_num_threads = this->num_neurons;
    int partitionSize = this->num_neurons / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = this->num_neurons - (tid*partitionSize);
        partitionStart = this->num_neurons - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    if(tid < new_num_threads) {
        for (int i = partitionStart; i < partitionStart + partitionSize; i++) {
            // this->Z[i] += this->b[i];
            this->Z[i][0] += this->b[i];
            this->A[i][0] = this->sigmoid(Z[i][0]);
        }
    }
}

void Layer::initializeGradients(int dZ_rows, int dZ_cols, int dW_rows, int dW_cols, int dB_rows, int dB_cols) {
    this->dZ_rows = dZ_rows;
    this->dZ_cols = dZ_cols;
    this->dW_rows = dW_rows;
    this->dW_cols = dW_cols;
    this->dB_rows = dB_rows;
    this->dB_cols = dB_cols;

    // this->dZ = new double*[dZ_rows];
    
    // for (int i = 0; i < dZ_rows; i++) {
    //     this->dZ[i] = new double[dZ_cols];
    // }
    this->dZ = this->allocate_2D(dZ_rows, dZ_cols);

    // this->dW = new double*[dW_rows];
    
    // for (int i = 0; i < dW_rows; i++) {
    //     this->dW[i] = new double[dW_cols];
    // }
    this->dW = this->allocate_2D(dW_rows, dW_cols);

    // this->dB = new double*[dB_rows];
    
    // for (int i = 0; i < dB_rows; i++) {
    //     this->dB[i] = new double[dB_cols];
    // }
    this->dB = this->allocate_2D(dB_rows, dB_cols);
}

void Layer::initializeWeights() {
    srand(1);
    for (int i = 0; i < this->num_neurons; i++) {
        for (int j = 0; j < this->num_input; j++) {
            double random = ((double) rand()) / (double) RAND_MAX;

            this->W[i][j] = -4.0 + (random * 8); 
        }
    }
}

void Layer::updateWeights(double lr, int tid, Barrier* barrier) {
    int new_num_threads1 = this->num_threads;
    int new_num_threads2 = this->num_threads;
    if(this->num_neurons < this->num_threads) new_num_threads1 = this->num_neurons;
    if(this->num_neurons < this->num_threads) new_num_threads2 = this->dB_rows;

    int partitionSize1 = this->num_neurons / new_num_threads1;
    int partitionStart1 = 0;
    int partitionSize2 = this->dB_rows / new_num_threads2;
    int partitionStart2 = 0;

    if(tid == new_num_threads1 - 1) {
        partitionSize1 = this->num_neurons - (tid*partitionSize1);
        partitionStart1 = this->num_neurons - partitionSize1;
    } else {
        partitionStart1 = tid * partitionSize1;
    }

    if(tid == new_num_threads2 - 1) {
        partitionSize2 = this->dB_rows - (tid*partitionSize2);
        partitionStart2 = this->dB_rows - partitionSize2;
    } else {
        partitionStart2 = tid * partitionSize2;
    }

    if(tid < new_num_threads1) {
        for (int i = partitionSize1; i < partitionSize1 + partitionStart1; i++) {
            for (int j = 0; j < this->num_input; j++) {
                this->W[i][j] -= lr * this->dW[i][j];
            }
        }
    }

    if(tid < new_num_threads2) {
        for (int i = partitionSize2; i < partitionSize2 + partitionStart2; i++) {
            this->b[i] -= lr * this->dB[i][0];
        }
    }

}


/***************** HELPER FUNCTIONS ********************************/

double** Layer::getActivations() {
    return this->A;
}

std::string Layer::getName() {
    return this->name;
}

int Layer::getNumInput() {
    return this->num_input;
}

int Layer::getNumNeurons() {
    return this->num_neurons;
}

double** Layer::allocate_2D(int rows, int cols) {
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

void Layer::free_2D(double** arr) {
    free(*arr);
    free(arr);
}

double** Layer::sigmoid_derivative(double* input_z, int input_length, int tid, Barrier* barrier) {
    
    if(tid == 0) {
        if(this->sigmoid_deriv_ret != NULL) this->free_2D(this->sigmoid_deriv_ret);
        this->sigmoid_deriv_ret = this->allocate_2D(input_length, 1);
    }
    

    int new_num_threads = this->num_threads;
    if(input_length < this->num_threads) new_num_threads = input_length;
    int partitionSize = input_length / new_num_threads;
    int partitionStart = 0;
    if(tid == new_num_threads - 1) {
        partitionSize = input_length - (tid*partitionSize);
        partitionStart = input_length - partitionSize;
    } else {
        partitionStart = tid * partitionSize;
    }

    
    barrier->barrier_exec(this->num_threads);

    if(tid < new_num_threads) {
        for (int i = partitionStart; i < partitionStart + partitionSize; i++) {
            double sig = this->sigmoid(input_z[i]);
            this->sigmoid_deriv_ret[i][0] = sig * (1 - sig);
        }
    }
    
    barrier->barrier_exec(this->num_threads);
    return this->sigmoid_deriv_ret; 
}

double Layer::sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}

/***************** DEBUG FUNCTIONS *********************************/

void Layer::initializeTestWeights() {
    int counter = 1;
    
    for (int i = 0; i < this->num_input; i++) {
        for (int j = 0; j < this->num_neurons; j++) {
            this->W[j][i] = counter;
            counter++;
        }
    }
}

void Layer::printA() {
    std::cout << "Printing layer " << this->name << " activations\n";
    for (int i = 0; i < this->num_neurons; i++) {
        printf("%lf ", this->A[i][0]);
        printf("\n");
    }
    // printf("\n");
}

void Layer::printLayerWeights() {
    std::cout << "Printing layer " << this->name << " weights\n";
    for (int i = 0; i < this->num_neurons; i++) {
        for (int j = 0; j < this->num_input; j++) {
            printf("%lf ", this->W[i][j]);
        }
        printf("\n");
    }
}

void Layer::printZ() {
    std::cout << "Printing layer " << this->name << " Z's\n";
    for (int i = 0; i < this->num_neurons; i++) {
        printf("%lf ", this->Z[i]);
    }
    printf("\n");
}

void Layer::printGradientSizes() {
    std::cout << "Printing layer " << this->name << " Gradient Sizes\n";
    printf("dZ: %d rows, %d cols\n", this->dZ_rows, this->dZ_cols);
    printf("dW: %d rows, %d cols\n", this->dW_rows, this->dW_cols);
    printf("dB: %d rows, %d cols\n", this->dB_rows, this->dB_cols);
}

void Layer::printdZ() {
    std::cout << "Printing layer " << this->name << " dZ's\n";
    for (int i = 0; i < this->dZ_rows; i++) {
        for (int j = 0; j < this->dZ_cols; j++) {
            printf("%lf ", this->dZ[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void Layer::printdW() {
    std::cout << "Printing layer " << this->name << " dW's\n";
    for (int i = 0; i < this->dW_rows; i++) {
        for (int j = 0; j < this->dW_cols; j++) {
            printf("%lf ", this->dW[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void Layer::printdB() {
    std::cout << "Printing layer " << this->name << " dB's\n";
    for (int i = 0; i < this->dB_rows; i++) {
        for (int j = 0; j < this->dB_cols; j++) {
            printf("%lf ", this->dB[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}