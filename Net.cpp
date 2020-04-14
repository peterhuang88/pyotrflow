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

//#include "Layer.h"
#include "Net.h"

Net::Net(double lr) {
    this->lr = lr;
    this->head = NULL;
    this->tail = NULL;
}

void Net::addLayer(int num_neurons) {
    if (head == NULL) {
        LayerNode* temp = (LayerNode*) malloc(sizeof(LayerNode));
        temp->curr = new Layer(5);
        temp->next = NULL;
        temp->prev = NULL;

        this->head = temp;
        this->tail = temp;
    } else {
        LayerNode* temp = (LayerNode*) malloc(sizeof(LayerNode));
        temp->curr = new Layer(5);
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