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
    this->input = new double*[input_size];
    for (int i = 0; i < input_size; i++) {
        this->input[i] = new double[1];
    }
    this->head = NULL;
    this->tail = NULL;
    this->label = 0;
}

Net::~Net() {
    for (int i = 0; i < this->input_size; i++) {
        delete [] input[i];
    }
    delete [] input;
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
    double** final_layer_act = temp->curr->getActivations();
    //temp->curr-> = final_layer_act[0][0] - this->label;
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

void Net::setInput(double* inp) {
    // make this faster with memcpy
    for (int i = 0; i < this->input_size; i++) {
        this->input[i][0] = inp[i];
    }
}

/***************** HELPER FUNCTIONS ********************************/



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