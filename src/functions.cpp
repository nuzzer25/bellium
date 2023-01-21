#include "functions.hpp"

// Sigmoid
MatrixXd Activation::sigmoid(MatrixXd &Z)
{ /* Sigmoid Activation Function */
    return 1.0 / (1.0 + (-Z).array().exp());
}

// Sigmoid Derivative
MatrixXd Activation::sigmoid_deriv(MatrixXd &Z)
{ /* Sigmoid Derivative Activation Function */
    return sigmoid(Z).array() * (1 - sigmoid(Z).array());
}

// Tanh
MatrixXd Activation::tanh(MatrixXd &Z)
{ /* Tanh Activation Function */
    return (Z.array().exp() - (-Z).array().exp()) / (Z.array().exp() + (-Z).array().exp());
}

// Tanh Derivative
MatrixXd Activation::tanh_deriv(MatrixXd &Z)
{ /* Tanh Derivative Activation Function */
    return Activation::tanh(Z).array().square();
}

// ReLU
MatrixXd Activation::ReLU(MatrixXd &Z)
{ /* ReLU Activation Function */
    return Z.cwiseMax(0);
}

// ReLU Derivative
MatrixXd Activation::ReLU_deriv(MatrixXd &Z)
{ /* ReLU Derivative Activation Function */
    MatrixXd y = Z;
    y.array() = (y.array() > 0).cast<double>();
    return y;
}
