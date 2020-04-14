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
        Layer(int num_neurons);
        ~Layer();

        std::string getName();
        
    private:
        double* weights;
        double* results;
        double* gradients;
        int num_neurons;
        std::string name; 
};

#endif