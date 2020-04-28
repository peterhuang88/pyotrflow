#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

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

struct thread_args { 
    int tid;
    double ** mat1; 
    int num_cols1;
    double ** mat2;
    int num_cols2;
    int partitionSize;
    int partitionStart;
    double ** res;
};

class Net {
    public:
        Net(double lr, int input_size, int numThreads);
        ~Net();
        
        // actually useful functions
        void addLayer(int num_input, int num_neurons, std::string name);
        void initializeGradients();
        void performBackProp(int tid);
        void performForwardProp(int tid);
        void setInput(double* inp, double label, int tid);
        void initializeNetWeights();
        void updateWeights();
        void trainNet(int num_epochs);
        double calculateLoss();

        // helper functions
        double** allocate_2D(int rows, int cols);
        void free_2D(double** arr);

        // parallel functions
        void * pTrain(void * args);
        
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
        pthread_t* threads;
        DatasetParser* parser;
        Barrier* barrier;
        int num_right;
        double cost;
        
};

#endif