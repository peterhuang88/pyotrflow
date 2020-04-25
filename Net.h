#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "Layer.h"

#ifndef NET_H
#define NET_H

struct LayerNode {
    Layer* curr;
    LayerNode* prev;
    LayerNode* next;
};

class Net {
    public:
        Net(double lr, int input_size);
        ~Net();
        
        // actually useful functions
        void addLayer(int num_input, int num_neurons, std::string name);
        void initializeGradients();
        void performBackProp();
        void performForwardProp();
        void setInput(double* inp);

        // helper functions

        // Debug Functions
        void printNet();
        void printNetActivations();
        void printNetWeights();
        void printGradientSizes();

        
        LayerNode* head;
        LayerNode* tail;

    private:
        double lr;
        int input_size;
        double** input;
        double label;
        double* prediction;
        
};

#endif