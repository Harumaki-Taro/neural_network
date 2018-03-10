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

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


//
// プロトタイプ宣言
//


void example(void) {
    Eigen::MatrixXf data(4,4);
    data << 5.1, 3.5, 1.4, 0.2,	//インスタンス1
            4.9, 3.0, 1.4, 0.2,	//インスタンス2
            6.2, 3.4, 5.4, 2.3, //インスタンス3
            5.9, 3.0, 5.1, 1.8;	//インスタンス4
        //    5.8, 2.8, 5.0, 2.0; //インスタンス5

    MatrixXf label(4,2);
    label << 1, 0,
             1, 0,
             0, 1,
        //     0, 1,
             0, 1;

    // Build a neural network archtecture.
    Neural_Network nn;
    int W1_shape[2] = { 4, 6 };
    nn.build_fullConnectedLayer(sigmoid, sigmoid_d, W1_shape, true);
    int W2_shape[2] = { 6, 3 };
    nn.build_fullConnectedLayer(sigmoid, sigmoid_d, W2_shape, true);
    int W3_shape[2] = { 3, 2 };
    nn.build_fullConnectedLayer(identity, identity_d, W3_shape, true);
    nn.build_outputLayer(2, softmax, "mean_cross_entropy");
    nn.allocate_memory(4);

    // Define loss function and optimizer.
    // Loss loss;
    // loss.add_crossEntropy();
    // // loss.add_LpNorm(2);
    // Train train(nn, loss);

    // Initialize the neural network and training environment.
    // train.build_updateTerms();

    // Define learning parameters.
    MatrixXf pred;
    unsigned int epoch = 1000;
    float eps = 1.f;

    for ( unsigned int i = 0; i != epoch; ++i ) {
        // train.update(data, label);
        pred = nn.forwardprop(data);
        nn.backprop(pred, label);

        for ( int i = 0; i != (int)nn.get_layers().size(); i++ ) {
            if ( nn.get_layers()[i]->get_trainable() ) {
                nn.get_layers()[i]->calc_differential(nn.get_layers()[i-1]->get_activated());
                nn.get_layers()[i]->bW = nn.get_layers()[i]->get_bW() - (eps * nn.get_layers()[i]->_dE_dbW.array()).matrix();
            }
        }
    }

    cout << "pred:" << endl;
    cout << nn.get_pred() << endl;
    cout << "cross_entropy_error" << endl;
    cout << nn.calc_loss_with_prev_pred(label) << endl;
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
