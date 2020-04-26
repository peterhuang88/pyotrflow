#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "Net.h"

Net::Net(double lr, int input_size) {
    this->lr = lr;
    this->input_size = input_size;
    // this->input = new double*[input_size];
    // for (int i = 0; i < input_size; i++) {
    //     this->input[i] = new double[1];
    // }
    this->input = this->allocate_2D(input_size, 1);
    this->head = NULL;
    this->tail = NULL;
    this->label = 0;
}

Net::~Net() {
    // for (int i = 0; i < this->input_size; i++) {
    //     delete [] input[i];
    // }
    // delete [] input;
    this->free_2D(this->input);
}

/***************** ACTUALLY USEFUL FUNCTIONS ***********************/

void Net::addLayer(int num_input, int num_neurons, std::string name) {
    if (head == NULL) {
        LayerNode* temp = (LayerNode*) malloc(sizeof(LayerNode));
        temp->curr = new Layer(num_input, num_neurons,1, name);
        temp->next = NULL;
        temp->prev = NULL;

        this->head = temp;
        this->tail = temp;
        // temp->curr->printLayerWeights();
    } else {
        LayerNode* temp = (LayerNode*) malloc(sizeof(LayerNode));
        temp->curr = new Layer(num_input, num_neurons, 0, name);
        temp->next = NULL;
        temp->prev = this->tail; // set new node's prev to the tail

        this->tail->next = temp; // set old tail's next to temp
        this->tail = temp;       // make temp the new tail
    }
}

void Net::initializeGradients() {
    // create temp layernode
    LayerNode* temp = this->tail;
    int prev_num_rows = temp->prev->curr->getNumNeurons();
    temp->curr->initializeGradients(1,1,1,prev_num_rows,1,1);

    temp = temp->prev;

    LayerNode* temp_prev;
    LayerNode* temp_next;
    int next_w_cols;
    int next_dz_cols;

    // keep going until we're at the last node
    while (temp->prev != NULL) {
        // get dimensions of weights matrix in next layer
        std::cout << temp->curr->name << " doing backprop\n";
        temp_prev = temp->prev;
        temp_next = temp->next;
        next_w_cols = temp_next->curr->getNumInput();
        next_dz_cols = temp_next->curr->dZ_cols;
        prev_num_rows = temp_prev->curr->getNumNeurons();


        temp->curr->initializeGradients(next_w_cols, next_dz_cols, next_w_cols, prev_num_rows, next_w_cols, next_dz_cols);

        temp = temp->prev;
    }

    // handle the last node
    temp_next = temp->next;
    next_w_cols = temp_next->curr->getNumInput();
    next_dz_cols = temp_next->curr->dZ_cols;

    temp->curr->initializeGradients(next_w_cols, next_dz_cols, next_w_cols, this->input_size, next_w_cols, next_dz_cols);
}

void Net::performBackProp() {
    // do final layer first
    LayerNode* temp = this->tail;
    double** A_prev = temp->prev->curr->getActivations();
    int A_prev_rows = temp->curr->num_input;
    int A_prev_cols = temp->curr->num_neurons;

    temp->curr->lastLayerBackProp(this->label, A_prev, A_prev_rows, A_prev_cols);


    
    LayerNode* temp_next;//  = temp->next;
    LayerNode* temp_prev;
    double** W_next;
    int W_next_rows;
    int W_next_cols;
    double** dZ_next;
    int dZ_next_rows;
    int dZ_next_cols;
    // now handle everything up until first layer
    while (temp->prev != NULL) {
        temp_next = temp->next;
        temp_prev = temp->prev;

        W_next = temp_next->curr->W;
        W_next_rows = temp_next->curr->num_neurons;
        W_next_cols = temp_next->curr->num_input;

        dZ_next = temp_next->curr->dZ;
        dZ_next_rows = temp_next->curr->dZ_rows;
        dZ_next_cols = temp_next->curr->dZ_cols;

        A_prev = temp->prev->curr->getActivations();
        A_prev_rows = temp->curr->num_input;
        A_prev_cols = temp->curr->num_neurons;

        temp->curr->backProp(W_next, W_next_rows, W_next_cols, dZ_next, dZ_next_rows, dZ_next_cols, A_prev, A_prev_rows, A_prev_cols);

        temp = temp->prev;
    }

}

void Net::performForwardProp() {
    double** A_prev = this->input;

    LayerNode* temp = this->head;

    while (temp != NULL) {
        // forward prop for given layer
        temp->curr->forwardProp(A_prev);
        A_prev = temp->curr->getActivations();
        temp = temp->next;
    }
}

void Net::setInput(double* inp, double label) {
    // make this faster with memcpy
    for (int i = 0; i < this->input_size; i++) {
        this->input[i][0] = inp[i];
    }

    this->label = label;
}

/***************** HELPER FUNCTIONS ********************************/
double** Net::allocate_2D(int rows, int cols) {
    int i;             /* Loop variable              */
    double **pointers; /* The pointers for each row  */
    double *array;     /* The actually array of ints */

    /* Allocate memory for the rows X cols array */
    array = (double *) malloc(rows * cols * sizeof(double));

    /* Allocate the array of pointers, one per row */
    pointers = (double**)malloc(rows * sizeof(double *)); 

    /* Point each pointer at its corresponding row */
    for (i = 0; i < rows; i++) {
        pointers[i] = array + (cols * i);
    }

    return pointers;
}

void Net::free_2D(double** arr) {
    free(*arr);
    free(arr);
}



/***************** DEBUG FUNCTIONS *********************************/

void Net::printNet() {
    LayerNode* temp = this->head;
    while (temp != NULL) {
        std::cout << temp->curr->getName() << ":\n";
        std::cout << "input size: " << temp->curr->getNumInput() << "\n";
        std::cout << "num neurons: " << temp->curr->getNumNeurons() << "\n";
        std::cout << "---------------------------\n";
        temp = temp->next;
    }
}

void Net::printNetActivations() {
    LayerNode* temp = this->head;
    while (temp != NULL) {
        std::cout << temp->curr->getName() << ":\n";
        // std::cout << "input size: " << temp->curr->getNumInput() << "\n";
        // std::cout << "num neurons: " << temp->curr->getNumNeurons() << "\n";
        temp->curr->printA();
        std::cout << "---------------------------\n";
        temp = temp->next;
    }
}

void Net::printNetWeights() {
    LayerNode* temp = this->head;
    while (temp != NULL) {
        std::cout << temp->curr->getName() << ":\n";
        // std::cout << "input size: " << temp->curr->getNumInput() << "\n";
        // std::cout << "num neurons: " << temp->curr->getNumNeurons() << "\n";
        temp->curr->printLayerWeights();
        std::cout << "---------------------------\n";
        temp = temp->next;
    }
}

void Net::printGradientSizes() {
    LayerNode* temp = this->head;
    while (temp != NULL) {
        std::cout << temp->curr->getName() << ":\n";
        // std::cout << "input size: " << temp->curr->getNumInput() << "\n";
        // std::cout << "num neurons: " << temp->curr->getNumNeurons() << "\n";
        temp->curr->printGradientSizes();
        std::cout << "---------------------------\n";
        temp = temp->next;
    }
}

void Net::printGradients() {
    LayerNode* temp = this->tail;
    printf("------------------Printing Net Gradients-------------------\n");
    while (temp != NULL) {
        std::cout << temp->curr->getName() << ":\n";
        // std::cout << "input size: " << temp->curr->getNumInput() << "\n";
        // std::cout << "num neurons: " << temp->curr->getNumNeurons() << "\n";
        // temp->curr->printGradientSizes();
        temp->curr->printdZ();
        temp->curr->printdW();
        temp->curr->printdB();


        std::cout << "---------------------------\n";
        temp = temp->prev;
        // break;
    }
}