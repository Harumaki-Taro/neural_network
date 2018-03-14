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


int main(void) {
    initialize();
    // Get dataset
    Mnist mnist = Mnist();

    // Define learning parameters.
    MatrixXf pred;
    float eps = 0.03;
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
    nn.allocate_memory(mini_batch_size, 28*28);

    // Define loss function and optimizer.
    // Loss loss;
    // loss.add_crossEntropy();
    // // loss.add_LpNorm(2);
    // Train train(nn, loss);

    // Initialize the neural network and training environment.
    // train.build_updateTerms();

    for ( unsigned int i = 0; i < epoch; i++ ) {
        std::chrono::system_clock::time_point  start, end;
        start = std::chrono::system_clock::now(); // 計測開始時間
        // train.update(data, label);
        Mini_Batch mini_batch = mnist._train.randomPop(mini_batch_size);
        pred = nn.forwardprop(mini_batch.example);
        nn.backprop(pred, mini_batch.label);

        for ( int j = 0; j != (int)nn.get_layers().size(); j++ ) {
            if ( nn.get_layers()[j]->get_trainable() ) {
                if ( nn.get_layers()[j]->get_type() == "full_connect_layer" ) {
                    nn.get_layers()[j]->bW[0][0]
                        = nn.get_layers()[j]->get_bW()[0][0]
                        - (eps * nn.get_layers()[j]->_dE_dbW[0][0].array()).matrix();
                } else if ( nn.get_layers()[j]->get_type() == "flatten_layer" ) {
                    ;
                } else if ( nn.get_layers()[j]->get_type() == "flatten_layer" ) {
                    ;
                } else if ( nn.get_layers()[j]->get_type() == "en_tensor_layer" ) {
                    ;
                } else if ( nn.get_layers()[j]->get_type() == "convolution_layer" ) {
                    for ( int k = 0; k < nn.get_layers()[j]->get_channel_num(); k++ ) {
                        for ( int l = 0; l < nn.get_layers()[j]->get_prev_channel_num(); l++ ) {
                            nn.get_layers()[j]->W[k][l]
                                = nn.get_layers()[j]->W[k][l]
                                - (eps * nn.get_layers()[j]->dE_dW[k][l].array()).matrix();
                        }
                    }
                    nn.get_layers()[j]->b
                        = nn.get_layers()[j]->b
                         - (eps * nn.get_layers()[j]->dE_db.array()).matrix();
                } else {
                    cout << "不詳クラス" << endl;
                    exit(1);
                }



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
        end = std::chrono::system_clock::now();  // 計測終了時間

        if ( i % 50 == 0 ) {
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
            cout << i << endl;
            cout << "loss: " << nn.calc_loss_with_prev_pred(mini_batch.label) << "\t(time: " << elapsed << "msec / example)" << endl;
        }

        if ( i + 1 == epoch ) {
            cout << pred << endl;
        }
    }

    return 0;
}


// namespace py = pybind11;
// PYBIND11_PLUGIN(neural_network) {
//     py::module m("neural_network", "neural_network made by pybind11");
//     m.def("example", &example);
//
//     return m.ptr();
// }
