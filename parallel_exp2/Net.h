#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>

#include "Layer.h"
#include "DatasetParser.h"
#include "Barrier.h"

#ifndef NET_H
#define NET_H

struct LayerNode {
    Layer* curr;
    LayerNode* prev;
    LayerNode* next;
};

class Net {
    public:
        Net(double lr, int input_size, int num_threads);
        ~Net();
        
        // actually useful functions
        void addLayer(int num_input, int num_neurons, std::string name);
        void initializeGradients();
        void performBackProp(int tid);
        void performForwardProp(int tid);
        void setInput(double* inp, double label, int tid);
        void initializeNetWeights();
        void updateWeights(int tid);
        void trainNet(int num_epochs);
        double calculateLoss();

        // helper functions
        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // parallel functions
        void * pTrain(int tid, int num_epochs);
        
        // Debug Functions
        void printNet();
        void printNetActivations();
        void printNetWeights();
        void printGradientSizes();
        void printGradients();

        
        LayerNode* head;
        LayerNode* tail;

    private:
        double lr;
        int input_size;
        double** input;
        double label;
        double* prediction;
        int num_threads;
        std::thread * threads;
        DatasetParser* parser;
        Barrier* barrier;
        int num_right;
        double cost;
        
};

#endif