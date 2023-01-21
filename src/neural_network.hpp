#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

// Main Header
#include "models.hpp"

// Neural Network Headers
#include "mlp.hpp"

// Creating Model
static Model *create(std::string type, py::handle layers, double learning_rate)
{
    // Converting Numpy Array or Python List to Vector
    std::vector<int> converted_layer;

    // Check if a Numpy Array
    if (py::isinstance<py::array>(layers))
    {
        py::array_t<int> arr = layers.cast<py::array_t<int>>();
        py::buffer_info buffer = arr.request();
        converted_layer.assign(static_cast<int *>(buffer.ptr), static_cast<int *>(buffer.ptr) + buffer.size);
    }
    // Check if a Python List
    else if (py::isinstance<py::list>(layers))
    {
        auto lst = layers.cast<py::list>();
        for (auto item : lst)
        {
            converted_layer.push_back(item.cast<int>());
        }
    }

    // Multi-Layer Perceptron
    if (type == "MLP")
    {
        return new MLP(converted_layer, learning_rate);
    }
    else
    {
        throw std::invalid_argument("Is An Invalid Architecture Type");
    }
}

#endif