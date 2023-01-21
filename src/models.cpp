#include "models.hpp"

// Using namespaces
using namespace std;

Model::Model(string type, vector<int> layers, double learning_rate)
{
    // Updating HyperParameters
    this->type = type;
    this->layers = layers;
    this->learning_rate = learning_rate;
}

void Model::attributes()
{
    cout << "\n"
         << "Neural Network: " << type << endl;
    cout << "Learning Rate: " << learning_rate << endl;
    cout << "Layers: ";
    for (int size : this->layers)
    {
        cout << size << " ";
    }
    cout << "\n"
         << endl;
}
