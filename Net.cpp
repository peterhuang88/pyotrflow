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
    this->input = new double[input_size];
    this->head = NULL;
    this->tail = NULL;
}

Net::~Net() {
    delete [] input;
}

void Net::addLayer(int num_neurons, int num_input, std::string name) {
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

void Net::printNet() {
    LayerNode* temp = this->head;
    while (temp != NULL) {
        std::cout << temp->curr->getName() << "\n";
        temp = temp->next;
    }
}