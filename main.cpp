#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>

#include "Layer.h"
#include "Net.h"
#include "DatasetParser.h"

int main(int argc, char** argv) {
    // Layer layer1(5);
    Net my_net(0.01, 3);
    //my_net.addLayer(3,3,"layer1");
    //my_net.addLayer(3,4,"layer2");
    //my_net.printNet();

    /*
    DatasetParser parser("./data/sonar.all-data", 0);

    // Use the below function to get size of input variables (60 for sonar data) or number of observations (208 for sonar data)
    std::cout << "Input variables: " << parser.getNumInputs() << std::endl;
    std::cout << "Observations: " << parser.getNumObservations() << std::endl;

    //Get output and input; getInput returns a double* of all of the input variables for the index
    std::cout << "First input variable of first observation: " << *(parser.getInput(0)) << std::endl;
    //Output is 0 for R, 1 for M
    std::cout << "First observation output: " << parser.getOutput(0) << std::endl; 

    // There are also functions that return the entirety of the output/input data (as vectors) or the input/output for a given index as a pair. See DatasetParser.cpp for methods.
    */ 
    double* input = new double[3];
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;
    my_net.setInput(input, 100);

    // Layer l1(3,4,1,"test_layer1");
    // l1.printLayerWeights();
    my_net.addLayer(3,4,"test_layer1");
    my_net.addLayer(4,2, "test_layer2");
    my_net.addLayer(2,1, "test_output");
    my_net.printNet();
    my_net.printNetWeights();
    my_net.performForwardProp();
    my_net.printNetActivations();

    my_net.initializeGradients();

    my_net.performBackProp();
    my_net.printGradients();
    //my_net.printGradientSizes();
    //l1.forwardProp(input);
    //l1.printZ();
    //l1.printA();


    std::cout << "Test done\n";
    return 0;
}