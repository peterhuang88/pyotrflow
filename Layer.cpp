#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>

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

    this->weights = new double*[num_input];
    for (int i = 0; i < num_input; i++) {
        this->weights[i] = new double[num_neurons];
    }
    this->results = new double[num_neurons];

    // TODO: THIS NEEDS TO CHANGE
    // this->gradients = new double[num_neurons];
    this->initializeTestWeights();
}

Layer::~Layer() {
    for (int i = 0; i < this->num_input; i++) {
        delete [] weights[i];
    }
    delete [] weights;
    delete [] results;
    //free(this->weights);
    //free(this->gradients);
    // free(this->results);
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Layer::forwardProp(double* input) {
    for (int i = 0; i < this->num_neurons; i++) {
        double sum = 0;
        for (int j = 0; j < this->num_input; j++) {
            sum += input[j] * this->weights[j][i];
        }
        this->results[i] = sum;
    }
}



/***************** HELPER FUNCTIONS ********************************/

std::string Layer::getName() {
    return this->name;
}

/***************** DEBUG FUNCTIONS *********************************/

void Layer::initializeTestWeights() {
    int counter = 1;
    for (int i = 0; i < this->num_input; i++) {
        for (int j = 0; j < this->num_neurons; j++) {
            this->weights[i][j] = counter;
            counter++;
        }
    }
}

void Layer::printLayerWeights() {
    std::cout << "Printing layer " << this->name << " weights\n";
    for (int i = 0; i < this->num_input; i++) {
        for (int j = 0; j < this->num_neurons; j++) {
            printf("%lf ", this->weights[i][j]);
        }
        printf("\n");
    }
}

void Layer::printResults() {
    std::cout << "Printing layer " << this->name << " results\n";
    for (int i = 0; i < this->num_neurons; i++) {
        printf("%lf ", this->results[i]);
    }
}