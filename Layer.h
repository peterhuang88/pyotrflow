#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "MatrixCalculator.h"

#ifndef LAYER_H
#define LAYER_H

class Layer {
    public:
        Layer(int num_input, int num_neurons, int marker, std::string name);
        ~Layer();

        // actually useful functions
        void backProp();
        void forwardProp(double** input);
        void initializeGradients(int dZ_rows, int dZ_cols, int dW_rows, int dW_cols, int dB_rows, int dB_cols);

        // random helper functions
        double** getActivations();
        std::string getName();
        int getNumInput();
        int getNumNeurons();

        // debug functions
        void initializeTestWeights();
        void printA();
        void printLayerWeights();
        void printZ();
        void printGradientSizes();

        int dZ_rows;
        int dZ_cols; 
        int dW_rows; 
        int dW_cols; 
        int dB_rows; 
        int dB_cols;
        
    private:
        int num_neurons;
        int num_input;
        std::string name; 
        int marker; // 0 = nothing special, 1 = head, 2 = tail
        MatrixCalculator mc;

        double** W; // this is actually W_t
        double* b; // bias
        double* Z; // wTx + b
        double** A; // A = activation_func(Z)
        double** dZ;
        double** dW;
        double** dB;

        

        // double* gradients;
};

#endif