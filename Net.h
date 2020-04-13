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
#endif
// #include "Layer.h"

struct LayerNode {
    Layer* curr;
    LayerNode* prev;
    LayerNode* next;
};

class Net {
    public:
        Net(double lr);
        // ~Net();
        
        void addLayer(int num_neurons);
        void printNet();
        
        LayerNode* head;
        LayerNode* tail;

    private:
        double lr;
};