#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>

#include "Layer.h"
#include "Net.h"

int main(int argc, char** argv) {
    // Layer layer1(5);
    Net my_net(0.01, 3);
    //my_net.addLayer(3,3,"layer1");
    //my_net.addLayer(3,4,"layer2");
    //my_net.printNet();
    double* input = new double[3];
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;
    my_net.setInput(input);

    // Layer l1(3,4,1,"test_layer1");
    // l1.printLayerWeights();
    my_net.addLayer(3,4,"test_layer1");
    my_net.addLayer(4,3, "test_layer2");
    // my_net.printNet();
    //my_net.printNetWeights();
    //my_net.performForwardProp();
    //my_net.printNetActivations();

    //l1.forwardProp(input);
    //l1.printZ();
    //l1.printA();


    std::cout << "Test done\n";
    return 0;
}