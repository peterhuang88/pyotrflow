#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "DatasetParser.h"

DatasetParser::DatasetParser(std::string dataset_path, int type) {
  this->dataset_path = dataset_path;
  this->type = type;

  this->parse();
}


// Parse function 
void DatasetParser::parse() {
  std::ifstream dataset(this->dataset_path);

  if(!dataset.good()) {
    std::cerr << "There was a problem with the dataset file - aborting" << std::endl;
    exit(1);
  } 

  if(this->type == 0) {
    // Sonar dataset
    while(dataset.good()) {
      std::string line;
      std::vector<double> result;

      std::getline(dataset, line);
      

      std::stringstream lineStream(line);
      std::string cell;

      while(std::getline(lineStream, cell, ',')) {
        if(cell.compare("R") != 0 && cell.compare("M") != 0) {
          result.push_back(atof(cell.c_str()));
        } else {    
          // R is 0, M is 1
          if(cell.compare("R") == 0) {
            this->outputs.push_back(0);
          } else {
            this->outputs.push_back(1);
          }
        }
      }

      this->inputs.push_back(result);
    }
    this->num_observations = this->outputs.size();
    this->num_inputs = this->inputs.empty() ? 0 : this->inputs.front().size();
  } else {
    std::cerr << "Dataset type unknown (must be 0) - aborting" << std::endl;
    exit(1);
  }
}

// Getters
int DatasetParser::getNumObservations() {
  return this->num_observations;
}

int DatasetParser::getNumInputs() {
  return this->num_inputs;
}

std::vector<std::vector<double> > DatasetParser::getAllInput() {
  return this->inputs;
}

std::vector<int> DatasetParser::getAllOutput() {
  return this->outputs;
}

double* DatasetParser::getInput(int index) {
  return this->inputs[index].data();
}

int DatasetParser::getOutput(int index) {
  return this->outputs[index];
}

std::pair<double*, int> DatasetParser::getObservation(int index) {
  return std::make_pair(this->inputs[index].data(), this->outputs[index]);
}
