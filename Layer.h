#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#ifndef LAYER_H
#define LAYER_H

class Layer {
    public:
        Layer(int num_input, int num_neurons, int marker, std::string name);
        ~Layer();

        // actually useful functions
        void forwardProp(double* input);

        // random helper functions
        double* getActivations();
        std::string getName();
        int getNumInput();
        int getNumNeurons();

        // debug functions
        void initializeTestWeights();
        void printA();
        void printLayerWeights();
        void printZ();
        
    private:
        int num_neurons;
        int num_input;
        std::string name; 
        int marker; // 0 = nothing special, 1 = head

        double** W; // this is actually W_t
        double b; // bias
        double* Z; // wTx + b
        double* A; // A = activation_func(Z)
        // double* gradients;
};

#endif