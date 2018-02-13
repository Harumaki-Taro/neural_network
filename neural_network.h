#include <list>
#include <functional>
#include <math.h>
#include "my_math.h"
#include <iostream>

using std::list;
using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
//
// training data
//


typedef struct {
    MatrixXf W;
    int W_shape[2];
    MatrixXf b;
    int b_shape;
    MatrixXf bW;
    int bW_shape[2];
    bool _use_bias;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    MatrixXf pre_activate;
    MatrixXf activated;
    MatrixXf activated_;
    MatrixXf delta;
    MatrixXf dE_dW;
    MatrixXf dE_dbW;
}Layer;


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
    list<Layer> Layers;
    int _batch_size;
    int _example_size;
};


Neural_Network::Neural_Network(void) {
    Layer input_layer;
    Layers.push_back(input_layer);
}


void Neural_Network::build_fullConnectedLayer(MatrixXf W, int W_rows, int W_columns,
                                             MatrixXf b, int b_rows, bool use_bias,
                                             function<MatrixXf(MatrixXf)> f,
                                             function<MatrixXf(MatrixXf)> d_f) {
    Layer layer;
    layer.W.resize(W_rows, W_columns);
    layer.W = W;
    layer.W_shape[0] = W_rows;
    layer.W_shape[1] = W_columns;
    layer.b.resize(b_rows, 1);
    layer.b = b;
    layer.b_shape = b_rows;
    layer._use_bias=use_bias;
    layer.f = f;
    layer.d_f = d_f;
    Layers.push_back(layer);
}


// void Neural_Network::build_softmaxLayer(void) {
//
// }


void Neural_Network::allocateMemory(int batch_size) {
    _batch_size = batch_size;
    auto fst_layer = ++Layers.begin();
    _example_size = fst_layer->W_shape[0];

    // input layer
    Layers.front().activated.resize(_batch_size, _example_size);
    Layers.front().activated_.resize(_batch_size, _example_size+1);
    if ( Layers.front()._use_bias ) {
        Layers.front().activated_.block(0,0,_batch_size,1) = MatrixXf::Ones(_batch_size, 1);
    } else {
        Layers.front().activated_.block(0,0,_batch_size,1) = MatrixXf::Zero(_batch_size, 1);
    }
    Layers.front().W_shape[0] = _batch_size;
    Layers.front().W_shape[1] = fst_layer->W_shape[0];

    // hidden layer
    for ( auto layer = ++Layers.begin(); layer != Layers.end(); layer++) {
        layer->pre_activate.resize(_batch_size, layer->W_shape[1]);
        layer->activated.resize(_batch_size, layer->W_shape[1]);
        layer->delta.resize(_batch_size, layer->W_shape[1]);
        layer->dE_dW.resize(layer->W_shape[0], layer->W_shape[1]);

        layer->bW.resize(layer->W_shape[0]+1, layer->W_shape[1]);
        layer->bW.block(0,0,1,layer->W_shape[1]) = layer->b;
        layer->bW.block(1,0,layer->W_shape[0],layer->W_shape[1]) = layer->W;
        layer->activated_.resize(_batch_size, layer->W_shape[1]+1);
        if ( layer->_use_bias ) {
            layer->activated_.block(0,0,_batch_size,1) = MatrixXf::Ones(_batch_size, 1);
        } else {
            layer->activated_.block(0,0,_batch_size,1) = MatrixXf::Zero(_batch_size, 1);
        }
        layer->dE_dbW.resize(layer->W_shape[0]+1, layer->W_shape[1]);
    }
    Layers.back().delta.resize(_batch_size, Layers.back().W_shape[1]);
}


MatrixXf Neural_Network::forwardprop(MatrixXf X) {
    Layers.front().activated_.block(0,1,_batch_size,_example_size) = X;

    for ( auto layer = Layers.begin(); layer != --Layers.end(); ) {
        auto prev_layer = layer;
        ++layer;
        layer->pre_activate = prev_layer->activated_ * layer->bW;
        layer->activated_.block(0,1,_batch_size,layer->W_shape[1]) = layer->f(layer->pre_activate);
    }
    MatrixXf pred = Layers.back().activated_.block(0,1,_batch_size,Layers.back().W_shape[1]);
    return pred;
}


void Neural_Network::backprop(MatrixXf y, MatrixXf pred) {
    MatrixXf pred_error = y - pred;
    Layers.back().delta = elemntwiseProduct(pred_error, Layers.back().d_f(pred));

    for ( auto layer = Layers.rbegin(); layer != --(--Layers.rend()); ) {
        auto next_layer = layer;
        ++layer;
        layer->delta = elemntwiseProduct(next_layer->delta * next_layer->bW.block(1,0,next_layer->W_shape[0],next_layer->W_shape[1]).transpose(),
                                         layer->d_f(layer->activated_.block(0,1,_batch_size,layer->W_shape[1])));
    }

    for ( auto layer = Layers.rbegin(); layer != --Layers.rend(); ) {
        auto next_layer = layer;
        ++layer;
        next_layer->dE_dbW = layer->activated_.transpose() * next_layer->delta;
        next_layer->bW = next_layer->bW + next_layer->dE_dbW;
    }
}
