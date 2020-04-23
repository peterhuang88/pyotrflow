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
    this->A = new double[num_neurons];
    this->b = 0.0;

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
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Layer::backProp() {
    
}

void Layer::forwardProp(double* input) {
    // perform wTx 
    /*
    for (int i = 0; i < this->num_neurons; i++) {
        double sum = 0;
        for (int j = 0; j < this->num_input; j++) {
            sum += this->W[i][j] * input[j];
        }
        this->Z[i] = sum + this->b;
    }*/
    mc.matrixTimesVector(this->W, num_neurons, num_input, input, num_neurons, this->Z);

    // calculate activations
    for (int i = 0; i < this->num_neurons; i++) {
        // this->A[i] = 1.0 / (1.0 + exp(-this->Z[i]));

        // TODO: comment out, this is only for testing
        this->A[i] = this->Z[i];
    }
}


/***************** HELPER FUNCTIONS ********************************/

double* Layer::getActivations() {
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
        printf("%lf ", this->A[i]);
    }
    printf("\n");
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
