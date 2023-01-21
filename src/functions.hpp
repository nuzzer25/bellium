#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

// External Headers
#include <Eigen/Dense>
#include <iostream>

// Namespace
using namespace Eigen;

// Activation Functions and Derivatives
class Activation
{
public:
    MatrixXd sigmoid(MatrixXd &);       // Sigmoid
    MatrixXd sigmoid_deriv(MatrixXd &); // Sigmoid Derivative
    MatrixXd tanh(MatrixXd &);          // Tanh
    MatrixXd tanh_deriv(MatrixXd &);    // Tanh Derivative
    MatrixXd ReLU(MatrixXd &);          // ReLU
    MatrixXd ReLU_deriv(MatrixXd &);    // ReLU Derivative
};

#endif