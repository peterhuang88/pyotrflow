#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <mpi.h>

#include "Layer.h"
#include "Net.h"
#include "DatasetParser.h"


int main(int argc, char** argv) {
    // Layer layer1(5);
    
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
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("world_size: %d, rank: %d\n", world_size, world_rank);
    
    double* input = new double[7];
    input[0] = 1;
    input[1] = -1;
    input[2] = 1;
    input[3] = 0.5;
    input[4] = 0.5;
    input[5] = 0.5;
    input[6] = 0.5;

    // my_net.setInput(input, 1);
    //Net my_net(0.01, 3, world_rank, world_size);
    //Net my_net(0.01, 3);
    //my_net.addLayer(3, 4, "test_hidden_layer");
    //my_net.addLayer(4,1, "output_layer");

    // my_net.mpiToy(world_rank);

    //my_net.printNet();

    // Layer l1(3,4,1,"test_layer1");
    // l1.printLayerWeights();
    

    Net my_net(0.01, 60, world_rank, world_size);
    // uncomment for 10 layer network
    my_net.addLayer(60,10,"test_layer1");
    my_net.addLayer(10,10,"test_layer2");
    my_net.addLayer(10,10,"test_layer3");
    my_net.addLayer(10,10,"test_layer4");
    my_net.addLayer(10,10,"test_layer5");
    my_net.addLayer(10,10, "test_hidden");
    my_net.addLayer(10,10, "test_hidden");
    my_net.addLayer(10,10, "test_hidden");
    my_net.addLayer(10,10, "test_hidden");
    my_net.addLayer(10,10, "test_hidden");
    my_net.addLayer(10,1,"output_layer"); 

    //my_net.addLayer(4,2, "test_layer2");
    //my_net.addLayer(2,1, "test_output");
    my_net.initializeNetWeights();
    //my_net.initializeNetTestWeights();

    //my_net.printNet();
    
    //my_net.performForwardProp();
    //my_net.printNetActivations();

    my_net.initializeGradients();

    //my_net.mpiToy(world_rank);
    //my_net.performForwardProp();
    //my_net.performBackProp();

    // my_net.printNetWeights();
    //my_net.syncNetWeights();

    struct timeval start, end;
    gettimeofday(&start, NULL);

    //my_net.performBackProp();
    //my_net.printGradients();
    //my_net.updateWeights();
    my_net.trainNet(10);
    //printf("-------------------POST SYNC------------------------\n");
    //my_net.printNetWeights();
    //my_net.printGradientSizes();
    //l1.forwardProp(input);
    //l1.printZ();
    //l1.printA();
    gettimeofday(&end, NULL); 

    double time_taken; 
  
    time_taken = (end.tv_sec - start.tv_sec) * 1e6; 
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6; 
  
    std::cout << "Time taken by program is : " << std::fixed << time_taken; 
    std::cout << " sec" << std::endl; 


    std::cout << "Test done\n";

    MPI_Finalize();
    return 0;
}