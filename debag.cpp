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
    unsigned int mini_batch_size = 50;

    // Build a neural network archtecture.
    Neural_Network nn;
    int W1_shape[2] = { 784, 500 };
    nn.add_layer( FullConnect_Layer(tanh_, tanh_d, 1, W1_shape) );
    int W2_shape[2] = { 500, 200 };
    nn.add_layer( FullConnect_Layer(tanh_, tanh_d, 1, W2_shape) );
    int W3_shape[2] = { 200, 10 };
    nn.add_layer( FullConnect_Layer(identity, identity_d, 1, W3_shape) );
    nn.add_layer( Output_Layer(softmax, mean_cross_entropy, diff, 10) );
    nn.allocate_memory(mini_batch_size, 28*28);

    // Neural_Network nn;
    // nn.add_layer( En_Tensor_Layer(1, 28, 28) );
    // nn.add_layer( Convolution_Layer(1, 2, 3, 3) );
    // nn.add_layer( Tensor_Activate_Layer(tanh_, tanh_d, 2)) ;
    // // nn.add_layer( Max_Pooling_Layer(2, 2, 2) );
    // nn.add_layer( Convolution_Layer(2, 5, 3, 3) );
    // nn.add_layer( Tensor_Activate_Layer(tanh_, tanh_d, 5)) ;
    // // nn.add_layer( Max_Pooling_Layer(5, 2, 2) );
    // nn.add_layer( Flatten_Layer(5, 24, 24) );
    // int W1_shape[2] = { 24*24*5, 500 };
    // nn.add_layer( FullConnect_Layer(tanh_, tanh_d, 1, W1_shape) );
    // int W2_shape[2] = { 500, 84 };
    // nn.add_layer( FullConnect_Layer(tanh_, tanh_d, 1, W2_shape) );
    // int W3_shape[2] = { 84, 10 };
    // nn.add_layer( FullConnect_Layer(identity, identity_d, 1, W3_shape) );
    // nn.add_layer( Output_Layer(softmax, mean_cross_entropy, diff, 10) );
    //
    // nn.allocate_memory(mini_batch_size, 28*28);

    // Define loss function and optimizer.
    for ( int i = 0; i < 10; i++ ) {
        Mini_Batch mini_batch = mnist._train.randomPop(mini_batch_size);
        nn.debag(mini_batch, 5, 100);
    }

    return 0;
}
