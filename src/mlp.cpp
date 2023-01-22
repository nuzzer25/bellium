#include "mlp.hpp"

// Using namespaces
using namespace std;

// Python List or Numpy Array
vector<int> conversion(py::handle layer)
{ /* The Function Converts a Python List or Numpy Array to a Vector */

    // Converting Numpy Array or Python List to Vector
    vector<int> converted_layer;

    // Check if a Numpy Array
    if (py::isinstance<py::array>(layer))
    {
        py::array_t<int> arr = layer.cast<py::array_t<int>>();
        py::buffer_info buffer = arr.request();
        converted_layer.assign(static_cast<int *>(buffer.ptr), static_cast<int *>(buffer.ptr) + buffer.size);
    }
    // Check if a Python List
    else if (py::isinstance<py::list>(layer))
    {
        auto lst = layer.cast<py::list>();
        for (auto item : lst)
        {
            converted_layer.push_back(item.cast<int>());
        }
    }
    return converted_layer;
}

// Zero Vector
vector<double> zero(int layer_size)
{
    vector<double> output;

    /* Creates a 1-D vector of zeros */
    for (int i = 0; i < layer_size; i++)
    {
        output.push_back(0);
    }
    return output;
}

// MLP Class Constructor
MLP::MLP(int input_layer, py::handle hidden_layers, int output_layer, double learning_rate)
{ /* Constructing the MLP */

    // Updating Parameters
    this->input_layer = input_layer;
    this->hidden_layers = conversion(hidden_layers);
    this->output_layer = output_layer;
    this->learning_rate = learning_rate;

    // Initialise
    init();
}

// MLP Attributes
void MLP::attributes()
{ /* MLP Attributes */

    cout << "\nLearning Rate: " << learning_rate << endl;

    cout << "Input Layer: " << this->input_layer << endl;

    cout << "Hidden Layers: ";
    for (int j : this->hidden_layers)
    {
        cout << j << " ";
    }
    cout << "\n"
         << endl;

    cout << "Output Layer: " << this->output_layer << endl;

    for (int i = 0; i < weights.size(); i++)
    {
        cout << "Weights of Layer " << i + 1 << ":\n";
        for (int j = 0; j < weights[i].size(); j++)
        {
            cout << "Weights of Neuron " << j + 1 << ": ";
            for (int k = 0; k < weights[i][j].size(); k++)
                cout << weights[i][j][k] << " ";
            cout << "\n";
        }
        cout << "Biases of Layer " << i + 1 << ": ";
        for (int j = 0; j < biases[i].size(); j++)
            cout << biases[i][j] << " ";
        cout << "\n\n";
    }
}

// Initialising Parameters
void MLP::init()
{ /* Initialising Parameters */

    // Iterating Over Layers
    for (int i = 0; i < hidden_layers.size() - 1; i++)
    {
        std::vector<std::vector<double>> layer_weights;
        std::vector<double> layer_biases;

        // Iterating Over Neurons
        for (int j = 0; j < hidden_layers[i + 1]; j++)
        {
            std::vector<double> neuron_weights;
            for (int k = 0; k < hidden_layers[i]; k++)
            {
                // Generates a number between 0 and 1
                double weight = (double)rand() / RAND_MAX;
                neuron_weights.push_back(weight);
            }
            double bias = (double)rand() / RAND_MAX;
            layer_weights.push_back(neuron_weights);
            layer_biases.push_back(bias);
        }
        // Updating the Weights and Biases
        weights.push_back(layer_weights);
        biases.push_back(layer_biases);
    }
}

// Forward Propagation
std::vector<double> MLP::forward_prop()
{ /* Foward Propagation */

    std::vector<double> output = zero(input_layer);

    // Iterating Over Weights
    for (int i = 0; i < weights.size(); i++)
    {
        std::vector<double> layer_output;
        for (int j = 0; j < weights[i].size(); j++)
        {
            double activation = biases[i][j];
            for (int k = 0; k < weights[i][j].size(); k++)
                activation += output[k] * weights[i][j][k];
            layer_output.push_back(sigmoid(activation));
        }
        output = layer_output;
    }
    cout << "Output of the MLP is : [ ";
    for (int i = 0; i < output.size(); i++)
        cout << output[i] << " ";
    cout << "]\n";
    return output;
}
