//
//  10lines_neural_network.cpp
//
// C++
// clang++ -std=c++14 neural_network.cpp
// Python
// c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` neural_network.cpp -o neural_network`python3-config --extension-suffix`
//
//

#include <iostream>
#include "Eigen/Core"
#include "my_math.h"
#include "neural_network.h"
// #include "loss.h"
// #include "train.h"
// #include <pybind11/pybind11.h>
#include "mnist.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


//
// プロトタイプ宣言
//


void example(void) {
    // Get dataset
    Mnist mnist = Mnist();

    // Define learning parameters.
    MatrixXf pred;
    float eps = 0.001;
    unsigned int mini_batch_size = 100;
    unsigned int epoch = 10000;

    // Build a neural network archtecture.
    Neural_Network nn;
    int W1_shape[2] = { 784, 500 };
    nn.build_fullConnectedLayer(tanh_, tanh_d, W1_shape, true);
    int W2_shape[2] = { 500, 200 };
    nn.build_fullConnectedLayer(tanh_, tanh_d, W2_shape, true);
    int W3_shape[2] = { 200, 10 };
    nn.build_fullConnectedLayer(identity, identity_d, W3_shape, true);
    nn.build_outputLayer(10, softmax, "mean_cross_entropy");
    nn.allocate_memory(mini_batch_size);

    // Define loss function and optimizer.
    // Loss loss;
    // loss.add_crossEntropy();
    // // loss.add_LpNorm(2);
    // Train train(nn, loss);

    // Initialize the neural network and training environment.
    // train.build_updateTerms();

    for ( unsigned int i = 0; i < epoch; i++ ) {
        // train.update(data, label);
        Mini_Batch mini_batch = mnist._train.randomPop(mini_batch_size);
        pred = nn.forwardprop(mini_batch.example);
        nn.backprop(pred, mini_batch.label);
        vector<MatrixXf> prev_bW{MatrixXf::Zero(1,1), MatrixXf::Zero(785,200), MatrixXf::Zero(201,10), MatrixXf::Zero(1,1)};

        for ( int j = 0; j != (int)nn.get_layers().size(); j++ ) {
            if ( nn.get_layers()[j]->get_trainable() ) {
                nn.get_layers()[j]->calc_differential(nn.get_layers()[j-1]->get_activated());
                nn.get_layers()[j]->bW
                    = nn.get_layers()[j]->get_bW()
                     - (eps * nn.get_layers()[j]->_dE_dbW.array()).matrix();

                // if ( i == 0 ) {
                //     nn.get_layers()[j]->calc_differential(nn.get_layers()[j-1]->get_activated());
                //     nn.get_layers()[j]->bW
                //         = nn.get_layers()[j]->get_bW()
                //          - (eps * nn.get_layers()[j]->_dE_dbW.array()).matrix();
                // } else if ( i == 1 ) {
                //     prev_bW[j] = nn.get_layers()[j]->bW;
                //
                //     nn.get_layers()[j]->calc_differential(nn.get_layers()[j-1]->get_activated());
                //     nn.get_layers()[j]->bW
                //         = nn.get_layers()[j]->get_bW()
                //          - (eps * nn.get_layers()[j]->_dE_dbW.array()).matrix();
                // } else {
                //     MatrixXf tmp = nn.get_layers()[j]->bW;
                //     MatrixXf diff = nn.get_layers()[j]->bW - prev_bW[j];
                //
                //     nn.get_layers()[j]->calc_differential(nn.get_layers()[j-1]->get_activated());
                //     nn.get_layers()[j]->bW
                //         = nn.get_layers()[j]->get_bW()
                //          - (eps * nn.get_layers()[j]->_dE_dbW.array()).matrix()
                //          + 0.9 * diff;
                //     prev_bW[j] = tmp;
                // }
            }
        }

        if ( i % 10 == 0 ) {
            cout << i << endl;
            cout << "loss: " << nn.calc_loss_with_prev_pred(mini_batch.label) << endl;
        }

        if ( i + 1 == epoch ) {
            cout << pred << endl;
        }
    }
}


int main(int argc, const char * argv[]) {
    std::srand((unsigned int) time(0));
    example();

	return 0;
}


// namespace py = pybind11;
// PYBIND11_PLUGIN(neural_network) {
//     py::module m("neural_network", "neural_network made by pybind11");
//     m.def("example", &example);
//
//     return m.ptr();
// }
