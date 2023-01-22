#include "functions.hpp"

// Sigmoid
double sigmoid(double x)
{ /* Sigmoid Activation Function */
    return 1 / (1 + exp(-x));
}

// Sigmoid Derivative
double sigmoid_deriv(double x)
{ /* Sigmoid Derivative Activation Function */
    return sigmoid(x) * (1 - sigmoid(x));
}
