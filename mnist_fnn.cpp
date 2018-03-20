//
//  mnist_fnn.cpp
//
// C++
// clang-omp++ -O3 -mtune=native -std=c++14 -fopenmp mnist_fnn.cpp
// Python
// c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` neural_network.cpp -o neural_network`python3-config --extension-suffix`
//
//

#include <iostream>
#include <chrono>
#include "Eigen/Core"
#include "initialize.h"
#include "my_math.h"
#include "neural_network.h"
#include "mnist.h"
#include "loss.h"
#include "sgd.h"
#include "momentum.h"
#include "train.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


//
// プロトタイプ宣言
//


int main(void) {
    initialize();
    // Get dataset
    Mnist mnist = Mnist();

    // Define learning parameters.
    MatrixXf pred;
    float learning_rate = 0.001;
    unsigned int mini_batch_size = 50;
    unsigned int epoch = 50000;

    // Build a neural network archtecture.
    Neural_Network nn;
    int W1_shape[2] = { 784, 500 };
    nn.add_layer( FullConnect_Layer(1, W1_shape) );
    nn.add_layer( Activate_Layer(relu, relu_d, 1) );
    int W2_shape[2] = { 500, 200 };
    nn.add_layer( FullConnect_Layer(1, W2_shape) );
    nn.add_layer( Activate_Layer(relu, relu_d, 1) );
    int W3_shape[2] = { 200, 10 };
    nn.add_layer( FullConnect_Layer(1, W3_shape) );
    nn.add_layer( Output_Layer(softmax, mean_cross_entropy, diff, 10) );
    nn.allocate_memory(mini_batch_size, 28*28);

    // Define loss function and optimizer.
    Loss loss(nn);
    loss.add_LpNorm(0.001, 2);

    SGD opt(learning_rate);
    Train train(loss, opt);

    for ( unsigned int i = 0; i < epoch; i++ ) {
        Mini_Batch mini_batch = mnist._train.randomPop(mini_batch_size);
        train.update(nn, mini_batch, i);
    }

    cout << nn.get_pred() << endl;

    return 0;
}
