#include "mlp.hpp"

// Using namespaces
using namespace std;

// MLP Class Constructor
MLP::MLP(vector<int> layers, double learning_rate) : Model("MLP", layers, learning_rate)
{
    // Updating HyperParameters
    this->layers = layers;
    this->learning_rate = learning_rate;
}
