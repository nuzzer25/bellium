#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

// External Headers
#include <Eigen/Dense>
#include <iostream>

// Namespace
using namespace Eigen;

// Activation Functions and Derivatives
double sigmoid(double);       // Sigmoid
double sigmoid_deriv(double); // Sigmoid Derivative

#endif