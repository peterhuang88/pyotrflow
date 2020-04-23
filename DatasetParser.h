#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifndef DATASET_PARSER_H
#define DATASET_PARSER_H

class DatasetParser {
    public:
        DatasetParser(std::string dataset_path, int type);

        // parse 
        void parse();
        
        // getters
        int getNumObservations();
        int getNumInputs();
        std::vector<std::vector<double> > getAllInput();
        std::vector<int> getAllOutput();
        double* getInput(int index);
        int getOutput(int index);
        std::pair<double*, int> getObservation(int index);
        
    private:
        std::string dataset_path;
        int type; // Type of dataset. 0 for sonar dataset (only dataset supported for now).
        int num_observations;
        int num_inputs;
        std::vector<std::vector<double> > inputs;
        std::vector<int> outputs;
};

#endif