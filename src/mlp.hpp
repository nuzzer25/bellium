#ifndef MLP_HPP
#define MLP_HPP

#include "models.hpp"

class MLP : public Model
{
private:
    // Dynamic HyperParameters
    std::vector<int> layers;                               // Layers as a vector
    std::vector<std::vector<std::vector<double>>> weights; // Weights
    std::vector<std::vector<double>> biases;               // Biases
    double learning_rate;                                  // Learning Rate

    // MLP Neural Network
    void forwardPropagation(std::vector<double> &); // Forward Propagation
    void backPropagation(std::vector<double> &);    // Backwards Propagation
    void gradientDescent();                         // Gradient of Descent
    void update();                                  // Updating Parameters

public:
    // MLP Constructor
    MLP(std::vector<int>, double);
};

#endif
