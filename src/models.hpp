#ifndef MODELS_HPP
#define MODELS_HPP

// Headers
#include <vector>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Namespace
namespace py = pybind11;

// Model Class
class Model
{
protected:
    std::string type;
    std::vector<int> layers;
    double learning_rate;

public:
    // Initialising Model
    Model(std::string, std::vector<int>, double);

    // Model Attributes
    void attributes();

    // Model Training
    // virtual void train(std::vector<std::vector<double>> &train_data, std::vector<std::vector<double>> &train_labels, int iteration) = 0;

    // Model Evaluation

    // Model Prediction
};

#endif