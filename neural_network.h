#ifndef INCLUDE_neural_network_h_
#define INCLUDE_neural_network_h_

#include <iostream>
#include <list>
#include <functional>
#include <math.h>
#include "Eigen/Core"
#include "full_connect_layer.h"

using std::list;
using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
//
// training data
//


class Neural_Network {
    /*
    */
public:
    // build layers
    void build_fullConnectedLayer(MatrixXf, int, int,
                                 MatrixXf, int, bool,
                                 function<MatrixXf(MatrixXf)>,
                                 function<MatrixXf(MatrixXf)>);
    // void build_softmaxLayer(void);

    // initialize for computing
    void allocateMemory(int);

    // train or evaluate
    MatrixXf forwardprop(MatrixXf);
    void backprop(MatrixXf, MatrixXf);

    // constructor
    Neural_Network(void);

private:
    list<FullConnect_Layer> Layers;
    int _batch_size;
    int _example_size;
};


Neural_Network::Neural_Network(void) {
    FullConnect_Layer input_layer;
    Layers.push_back(input_layer);
}


void Neural_Network::build_fullConnectedLayer(MatrixXf W, int W_rows, int W_columns,
                                             MatrixXf b, int b_rows, bool use_bias,
                                             function<MatrixXf(MatrixXf)> f,
                                             function<MatrixXf(MatrixXf)> d_f) {
    FullConnect_Layer layer;
    layer.build_layer(b, W, use_bias, f, d_f);
    Layers.push_back(layer);
}


// void Neural_Network::build_softmaxLayer(void) {
//
// }


void Neural_Network::allocateMemory(int batch_size) {
    _batch_size = batch_size;
    auto fst_layer = ++Layers.begin();
    _example_size = fst_layer->get_W().rows();

    // input layer
    Layers.front().activated_.resize(_batch_size, _example_size+1);
    if ( fst_layer->get_use_bias() ) {
        Layers.front().activated_.block(0,0,_batch_size,1) = MatrixXf::Ones(_batch_size, 1);
    } else {
        Layers.front().activated_.block(0,0,_batch_size,1) = MatrixXf::Zero(_batch_size, 1);
    }
    Layers.front().W.resize(_batch_size, fst_layer->W.rows());

    // hidden layer
    for ( auto layer = ++Layers.begin(); layer != Layers.end(); ) {
        auto prev_layer = layer;
        layer++;
        if ( layer != Layers.end() ) {
            prev_layer->allocate_memory(_batch_size, *layer);
        } else {
            prev_layer->allocate_memory(_batch_size);
        }
    }
}


MatrixXf Neural_Network::forwardprop(MatrixXf X) {
    Layers.front().activated_.block(0,1,_batch_size,_example_size) = X;

    for ( auto layer = Layers.begin(); layer != --Layers.end(); ) {
        auto prev_layer = layer;
        ++layer;
        layer->forwardprop(prev_layer->activated_);
    }
    MatrixXf pred = Layers.back().activated_.block(0,1,_batch_size,Layers.back().W.cols());
    return pred;
}


void Neural_Network::backprop(MatrixXf y, MatrixXf pred) {
    MatrixXf pred_error = y - pred;
    Layers.back().delta = elemntwiseProduct(pred_error, Layers.back().d_f(pred));

    for ( auto layer = Layers.rbegin(); layer != --(--Layers.rend()); ) {
        auto next_layer = layer;
        ++layer;
        layer->calc_delta(*next_layer);
    }

    for ( auto layer = Layers.rbegin(); layer != --Layers.rend(); ) {
        auto next_layer = layer;
        ++layer;
        next_layer->calc_differential(*layer);
        next_layer->bW = next_layer->bW + next_layer->dE_dbW;
    }
}


#endif // INCLUDE_neural_network_h_
