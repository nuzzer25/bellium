#ifndef MLP_HPP
#define MLP_HPP

// Headers
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "functions.hpp"

// Namespace
namespace py = pybind11;

class MLP
{
private:
    // Dynamic Parameters
    int input_layer;                                       // Number of Hidden Layers
    std::vector<int> hidden_layers;                        // Number of Hidden Layers
    int output_layer;                                      // Number of Hidden Layers
    double learning_rate;                                  // Learning Rate
    std::vector<std::vector<std::vector<double>>> weights; // Weights
    std::vector<std::vector<double>> biases;               // Biases

public:
    // MLP Constructor
    MLP(int, py::handle, int, double);

    // MLP Neural Network
    void attributes();                  // Attributes
    void init();                        // Initialise Parameters
    std::vector<double> forward_prop(); // Forward Propagation
    // std::vector<double> MLP::backward_prop(std::vector<double>); // Backward Propagation
    // void gradientDescent();                                      // Gradient of Descent
    // void update();                                               // Updating Parameters
};

#endif
