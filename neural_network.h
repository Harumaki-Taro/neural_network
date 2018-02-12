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
    // MatrixXf b;
    // int b_shape;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    MatrixXf pre_activate;
    MatrixXf activated;
    MatrixXf W_delta;
    MatrixXf dE_dW;
}Layer;


class Neural_Network {
    /*
    */
public:
    void buildFullConnectedLayer(MatrixXf, int, int,
                                 // MatrixXf, int,
                                 function<MatrixXf(MatrixXf)>,
                                 function<MatrixXf(MatrixXf)>);
    void allocateMemory(int);
    MatrixXf forwardprop(MatrixXf, int, int);
    void backprop(MatrixXf, MatrixXf, int);
    Neural_Network(void);

private:
    list<Layer> Layers;
};


Neural_Network::Neural_Network(void) {
    Layer input_layer;
    Layers.push_back(input_layer);
}


void Neural_Network::buildFullConnectedLayer(MatrixXf W, int W_rows, int W_columns,
                                             // MatrixXf b, int b_rows,
                                             function<MatrixXf(MatrixXf)> f,
                                             function<MatrixXf(MatrixXf)> d_f) {
    Layer layer;
    layer.W.resize(W_rows, W_columns);
    layer.W = W;
    layer.W_shape[0] = W_rows;
    layer.W_shape[1] = W_columns;
    // layer.b.resize(b_rows, 1);
    // layer.b = b;
    // layer.b_shape = b_rows;
    layer.f = f;
    layer.d_f = d_f;
    Layers.push_back(layer);
}


void Neural_Network::allocateMemory(int batch_size) {
    auto fst_layer = ++Layers.begin();
    Layers.front().activated.resize(batch_size, fst_layer->W_shape[0]);

    for ( auto layer = ++Layers.begin(); layer != Layers.end(); ++layer) {
        layer->pre_activate.resize(batch_size, layer->W_shape[1]);
        layer->activated.resize(batch_size, layer->W_shape[1]);
        layer->W_delta.resize(batch_size, layer->W_shape[1]);
        layer->dE_dW.resize(layer->W_shape[0], layer->W_shape[1]);
    }
}


MatrixXf Neural_Network::forwardprop(MatrixXf X, int X_rows, int X_columns) {
    Layers.begin()->activated = X;
    Layers.begin()->W_shape[0] = X_rows;
    Layers.begin()->W_shape[1] = X_columns;

    for ( auto layer = Layers.begin(); layer != --Layers.end(); ) {
        auto prev_layer = layer;
        ++layer;
        layer->pre_activate = prev_layer->activated * layer->W;
        layer->activated = layer->f(layer->pre_activate);
    }
    MatrixXf pred = Layers.back().activated;

    return pred;
}


void Neural_Network::backprop(MatrixXf y, MatrixXf pred, int batch_size) {
    MatrixXf pred_error = y - pred;
    Layers.back().W_delta = elemntwiseProduct(pred_error, Layers.back().d_f(pred));

    for ( auto layer = Layers.rbegin(); layer != --(--Layers.rend()); ) {
        auto next_layer = layer;
        ++layer;
        layer->W_delta = elemntwiseProduct(next_layer->W_delta * next_layer->W.transpose(), layer->d_f(layer->activated));
    }

    for ( auto layer = Layers.rbegin(); layer != --Layers.rend(); ) {
        auto next_layer = layer;
        ++layer;
        next_layer->dE_dW = layer->activated.transpose() * next_layer->W_delta;
        next_layer->W = next_layer->W + next_layer->dE_dW;
    }
}
