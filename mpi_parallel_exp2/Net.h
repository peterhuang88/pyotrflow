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
#include "DatasetParser.h"

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
        Net(double lr, int input_size, int world_rank, int world_size);
        ~Net();
        
        // actually useful functions
        void addLayer(int num_input, int num_neurons, std::string name);
        void initializeGradients();
        void performBackProp();
        void performForwardProp();
        void setInput(double* inp, double label);
        void initializeNetWeights();
        void initializeNetTestWeights();
        void updateWeights();
        void trainNet(int num_epochs);
        double calculateLoss();
        void syncNetWeights();

        // helper functions
        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // Debug Functions
        void printNet();
        void printNetActivations();
        void printNetWeights();
        void printGradientSizes();
        void printGradients();
        void mpiToy(int world_rank);

        
        LayerNode* head;
        LayerNode* tail;
        double* stuff;
        int world_rank;
        int world_size;
        int mpi_on;
        int num_train_examples;

    private:
        double lr;
        int input_size;
        double** input;
        double label;
        double* prediction;
        DatasetParser* parser;
        
};

#endif