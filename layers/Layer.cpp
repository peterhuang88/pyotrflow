#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "Layer.h"

Layer::Layer(int num_neurons) {
    this->num_neurons = num_neurons;
    this->weights = (double*) calloc(num_neurons, sizeof(double));
    this->results = (double*) calloc(num_neurons, sizeof(double));
    this->gradients = (double*) calloc(num_neurons, sizeof(double));
    this->name = "fully_connected";
}

Layer::~Layer() {
    free(this->weights);
    free(this->gradients);
    free(this->results);
}