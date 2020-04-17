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
        std::string getName();

        // debug functions
        void initializeTestWeights();
        void printLayerWeights();
        void printResults();
        
    private:
        int num_neurons;
        int num_input;
        std::string name; 
        int marker; // 0 = nothing special, 1 = head

        double** weights;
        double* results;
        // double* gradients;
};

#endif