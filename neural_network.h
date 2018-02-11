#include <vector>
#include <list>
#include <functional>
#include <math.h>
#include "my_math.h"
#include <iostream>

using std::vector;
using std::list;
using std::function;
using std::cout;
using std::endl;
//
// training data
//

typedef struct {
    vector<float> _;
    vector<int> shape;
} Array;


typedef struct {
    vector<float> W;
    int W_shape[2];
    vector<float> b;
    int b_shape;
    function<vector<float>(vector<float>)> f;
    function<vector<float>(vector<float>)> d_f;
    vector<float> pre_activate;
    vector<float> activated;
    vector<float> W_delta;
    vector<float> dE_dW;
}Layer;


class Neural_Network {
    /*
    */
public:
    void buildFullConnectedLayer(vector<float>, int, int,
                                 // vector<float>, int,
                                 function<vector<float>(vector<float>)>,
                                 function<vector<float>(vector<float>)>);
    vector<float> forwardprop(vector<float>, int, int);
    void backprop(vector<float>, vector<float>, int);
    Neural_Network(void);

private:
    list<Layer> Layers;
};


Neural_Network::Neural_Network(void) {
    Layer input_layer;
    Layers.push_back(input_layer);
}


void Neural_Network::buildFullConnectedLayer(vector<float> W, int W_rows, int W_columns,
                                             // vector<float> b, int b_rows,
                                             function<vector<float>(vector<float>)> f,
                                             function<vector<float>(vector<float>)> d_f) {
    Layer layer;
    layer.W = W;
    layer.W_shape[0] = W_rows;
    layer.W_shape[1] = W_columns;
    // layer.b = b;
    // layer.b_shape = b_rows;
    layer.f = f;
    layer.d_f = d_f;
    Layers.push_back(layer);
}


vector<float> Neural_Network::forwardprop(vector<float> X, int X_rows, int X_columns) {
    Layers.begin()->activated = X;
    Layers.begin()->W_shape[0] = X_rows;
    Layers.begin()->W_shape[1] = X_columns;

    for ( auto layer = Layers.begin(); layer != --Layers.end(); ) {
        auto prev_layer = layer;
        ++layer;
        layer->pre_activate = dot(prev_layer->activated, layer->W, prev_layer->W_shape[0], prev_layer->W_shape[1], layer->W_shape[1]);
        layer->activated = layer->f(layer->pre_activate);
    }
    vector<float> pred = Layers.back().activated;

    return pred;
}


void Neural_Network::backprop(vector<float> y, vector<float> pred, int batch_size) {
    vector<float> pred_error = y - pred;
    Layers.back().W_delta = pred_error * Layers.back().d_f(pred);

    for ( auto layer = Layers.rbegin(); layer != --(--Layers.rend()); ) {
        auto next_layer = layer;
        ++layer;
        layer->W_delta = dot(next_layer->W_delta, transpoose(next_layer->W, next_layer->W_shape[0], next_layer->W_shape[1]), batch_size, next_layer->W_shape[1], next_layer->W_shape[0]) * layer->d_f(layer->activated);
    }

    for ( auto layer = Layers.rbegin(); layer != --Layers.rend(); ) {
        auto next_layer = layer;
        ++layer;
        next_layer->dE_dW = dot(transpoose(layer->activated, batch_size, next_layer->W_shape[0]), next_layer->W_delta, next_layer->W_shape[0], batch_size, next_layer->W_shape[1]);
        next_layer->W = next_layer->W + next_layer->dE_dW;
    }
}
