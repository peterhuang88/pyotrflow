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
        // void backProp();
        void backProp(double** W_next, int W_next_rows, int W_next_cols, double** dZ_next, int dZ_next_rows, int dZ_next_cols, double** A_prev, int A_prev_rows, int A_prev_cols);
        void forwardProp(double** input);
        void lastLayerBackProp(double Y, double** A_prev, int A_prev_rows, int A_prev_cols);
        void initializeGradients(int dZ_rows, int dZ_cols, int dW_rows, int dW_cols, int dB_rows, int dB_cols);

        // random helper functions
        double** getActivations();
        std::string getName();
        int getNumInput();
        int getNumNeurons();

        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        double** sigmoid_derivative(double* input_z, int input_length);
        double sigmoid(double input);

        // debug functions
        void initializeTestWeights();
        void printA();
        void printLayerWeights();
        void printZ();
        void printGradientSizes();
        void printdZ();
        void printdW();
        void printdB();

        int dZ_rows;
        int dZ_cols; 
        int dW_rows; 
        int dW_cols; 
        int dB_rows; 
        int dB_cols;
        double** dZ;
        double** dW;
        double** dB;
        int num_neurons;
        int num_input;
        double** W; // this is actually W_t

    private:
        
        std::string name; 
        int marker; // 0 = nothing special, 1 = head, 2 = tail
        MatrixCalculator mc;

        
        double* b; // bias
        double* Z; // wTx + b
        double** A; // A = activation_func(Z)
        // double** dZ;
        // double** dW;
        // double** dB;

        

        // double* gradients;
};

#endif