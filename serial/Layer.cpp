#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>


#include "Layer.h"


/**
* num_neurons = number of neurons in current layer
* num_input = number of inputs from previous layer e.g. number of neurons in previous layer
*/
Layer::Layer(int num_input, int num_neurons, int marker, std::string name) {
    this->num_neurons = num_neurons;
    this->num_input = num_input;
    this->name = name;
    this->marker = marker;

    // this->W = new double*[num_neurons];
    // for (int i = 0; i < num_neurons; i++) {
    //     this->W[i] = new double[num_input];
    // }
    this->W = this->allocate_2D(num_neurons, num_input);

    this->Z = new double[num_neurons];

    // this->A = new double*[num_neurons];
    // // TODO only works for fc layers
    // for (int i = 0; i < num_neurons; i++) {
    //     this->A[i] = new double[1];
    // }
    this->A = this->allocate_2D(num_neurons, 1);

    
    this->b = new double[num_neurons];

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
    delete [] this->Z;
    free_2D(this->A);
    delete [] this->b;
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Layer::backProp(double** W_next, int W_next_rows, int W_next_cols, double** dZ_next, int dZ_next_rows, int dZ_next_cols, double** A_prev, int A_prev_rows, int A_prev_cols) {
    // calculate dZ
    double** W_next_transpose = mc.transposeMatrix(W_next, W_next_rows, W_next_cols);
    double** temp1 = mc.matrixTimesMatrix(W_next_transpose, W_next_cols, W_next_rows, dZ_next, dZ_next_rows, dZ_next_cols);

    double** deriv = this->sigmoid_derivative(this->Z, this->num_neurons);
    mc.hadamardProduct(temp1, deriv, this->num_neurons, 1, this->dZ);

}

void Layer::lastLayerBackProp(double Y, double** A_prev, int A_prev_rows, int A_prev_cols) {
    this->dZ[0][0] = this->A[0][0] - Y;
    double** A_prev_transpose = mc.transposeMatrix(A_prev, A_prev_rows, A_prev_cols);
    //this->free_2D(this->dW);
    this->dW = mc.matrixTimesMatrix(this->dZ, this->dZ_rows, this->dZ_cols, A_prev_transpose, A_prev_cols, A_prev_rows);
    //this->free_2D(this->dB);
    this->dB = this->dZ;

    // deallocate A_prev_transpose because we made alloc'd a new array for it
    //this->free_2D(A_prev_transpose);
}

void Layer::forwardProp(double** input) {
    // perform wTx 
    
    mc.matrixTimesVector(this->W, num_neurons, num_input, input, num_neurons, this->Z);
    // add bias to each z
    // TODO: potentially parallelize
    for (int i = 0; i < this->num_neurons; i++) {
        this->Z[i] += this->b[i];
    }

    // calculate activations
    for (int i = 0; i < this->num_neurons; i++) {
        // this->A[i] = 1.0 / (1.0 + exp(-this->Z[i]));

        // TODO: comment out, this is only for testing
        // this->A[i][0] = this->Z[i];
        this->A[i][0] = this->sigmoid(Z[i]);
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

void Layer::updateWeights(double lr) {
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_input; j++) {
            this->W[i][j] -= lr * this->dW[i][j];
        }
    }

    for (int i = 0; i < dB_rows; i++) {
        this->b[i] -= lr * this->dB[i][0];
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

double** Layer::sigmoid_derivative(double* input_z, int input_length) {
    double** ret = this->allocate_2D(input_length, 1);

    for (int i = 0; i < input_length; i++) {
        double sig = this->sigmoid(input_z[i]);
        ret[i][0] = sig * (1 - sig);
    }

    return ret; 
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