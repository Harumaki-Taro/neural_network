//
//  mnist_cnn.cpp
//
// C++
// clang-omp++ -O3 -mtune=native -std=c++14 -fopenmp mnist_cnn.cpp
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
    float learning_rate = 0.01;
    unsigned int mini_batch_size = 50;
    unsigned int epoch = 50000;

    // Build a neural network archtecture.
    Neural_Network nn;
    nn.add_layer( En_Tensor_Layer(1, 28, 28) );
    nn.add_layer( Convolution_Layer(tanh_, tanh_d, 1, 2, 3, 3) );
    // nn.add_layer( Max_Pooling_Layer(2, 2, 2) );
    nn.add_layer( Convolution_Layer(tanh_, tanh_d, 2, 2, 3, 3) );
    // nn.add_layer( Max_Pooling_Layer(2, 2, 2) );
    nn.add_layer( Flatten_Layer(2, 24, 24) );
    // int W1_shape[2] = { 2*24*24, 500 };
    // nn.add_layer( FullConnect_Layer(tanh_, tanh_d, W1_shape) );
    // int W2_shape[2] = { 500, 200 };
    // nn.add_layer( FullConnect_Layer(tanh_, tanh_d, W2_shape) );
    int W3_shape[2] = { 2*24*24, 10 };
    nn.add_layer( FullConnect_Layer(identity, identity_d, W3_shape) );
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
