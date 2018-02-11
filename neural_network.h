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


vector<float> W1_ {
     0.5,  0.5,  0.5, -0.5,  0.1,  0.5,
     0.5,  0.1, -0.5,  0.5,  0.5,  0.1,
     0.5, -0.5,  0.1,  0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,  0.1,  0.5,  0.5
};

vector<float> W2_ {
	 0.1,  0.5, -0.5,
	 0.5, -0.1,  0.5,
	-0.5,  0.5,  0.1,
	 0.1,  0.5, -0.5,
	 0.5, -0.1,  0.5,
	-0.5,  0.5,  0.1
};


vector<float> W3_ {
	0.1, 0.5,
	0.5, 0.1,
	0.1, 0.5
};


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
    vector<float> forwardprop(vector<float>, int, int);
    void backprop(vector<float>, vector<float>, int);
    // void build_Layer(vector<float>, vector<float>, int, int, int, function<vector<float>>);
    Neural_Network() {
        Layer1.W = W1_;
        Layer1.W_shape[0] = 4;
        Layer1.W_shape[1] = 6;
        Layer1.f = sigmoid;
        Layer1.d_f = sigmoid_d;

        Layer2.W = W2_;
        Layer2.W_shape[0] = 6;
        Layer2.W_shape[1] = 3;
        Layer2.f = sigmoid;
        Layer2.d_f = sigmoid_d;

        Layer3.W = W3_;
        Layer3.W_shape[0] = 3;
        Layer3.W_shape[1] = 2;
        Layer3.f = sigmoid;
        Layer3.d_f = sigmoid_d;

        Layers.push_back(Layer0);
        Layers.push_back(Layer1);
        Layers.push_back(Layer2);
        Layers.push_back(Layer3);
    };

private:
    list<Layer> Layers;

    Layer Layer0;
    Layer Layer1;
    Layer Layer2;
    Layer Layer3;
};


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


// void Neural_Network::build_Layer(vector<float> pre_L, vector<float> W, int pre_L_row,
//                                         int pre_L_col, int W_col, function<vector<float>> f) {
//     vector<float> pre_activate = dot(pre_L, W, pre_L_row, pre_L_col, W_col);
//     vector<float> activated = f(pre_activate);
//     forward_list.push_back(activated);
//
//     BackUnit backward;
//     backward.delta = dot(W2_delta, transpoose(W2, 6, 3), 4, 3, 6) * sigmoid_d(L1);
// }
