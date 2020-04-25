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

    this->W = new double*[num_neurons];
    for (int i = 0; i < num_neurons; i++) {
        this->W[i] = new double[num_input];
    }
    this->Z = new double[num_neurons];

    this->A = new double*[num_neurons];
    // TODO only works for fc layers
    for (int i = 0; i < num_neurons; i++) {
        this->A[i] = new double[1];
    }

    for (int i = 0; i < num_neurons; i++) {
        this->W[i] = new double[num_input];
    }
    this->b = new double[num_neurons];

    // TODO: THIS NEEDS TO CHANGE
    // this->gradients = new double[num_neurons];
    this->initializeTestWeights();
}

Layer::~Layer() {
    for (int i = 0; i < this->num_input; i++) {
        delete [] W[i];
    }
    delete [] W;
    delete [] Z;
    delete [] b;
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Layer::backProp() {
    
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
        this->A[i][0] = this->Z[i];
    }
}

void Layer::initializeGradients(int dZ_rows, int dZ_cols, int dW_rows, int dW_cols, int dB_rows, int dB_cols) {
    this->dZ_rows = dZ_rows;
    this->dZ_cols = dZ_cols;
    this->dW_rows = dW_rows;
    this->dW_cols = dW_cols;
    this->dB_rows = dB_rows;
    this->dB_cols = dB_cols;

    this->dZ = new double*[dZ_rows];
    
    for (int i = 0; i < dZ_rows; i++) {
        this->dZ[i] = new double[dZ_cols];
    }

    this->dW = new double*[dW_rows];
    
    for (int i = 0; i < dW_rows; i++) {
        this->dW[i] = new double[dW_cols];
    }

    this->dB = new double*[dB_rows];
    
    for (int i = 0; i < dW_rows; i++) {
        this->dB[i] = new double[dB_cols];
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
