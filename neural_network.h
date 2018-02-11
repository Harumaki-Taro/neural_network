#include <vector>
#include <list>
#include <functional>
#include <math.h>
#include "my_math.h"

using std::vector;
using std::list;
using std::function;
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
    };

private:
    list<vector<float>> forward_list;
    list<vector<float>> backward_list;

    Layer Layer0;
    Layer Layer1;
    Layer Layer2;
    Layer Layer3;
};

vector<float> Neural_Network::forwardprop(vector<float> X, int X_rows, int X_columns) {
    Layer0.activated = X;
    Layer0.W_shape[0] = X_rows;
    Layer0.W_shape[1] = X_columns;

    Layer1.pre_activate = dot(Layer0.activated, Layer1.W, Layer0.W_shape[0], Layer0.W_shape[1], Layer1.W_shape[1]);
    Layer1.activated = Layer1.f(Layer1.pre_activate);

    Layer2.pre_activate = dot(Layer1.activated, Layer2.W, Layer1.W_shape[0], Layer1.W_shape[1], Layer2.W_shape[1]);
    Layer2.activated = Layer2.f(Layer2.pre_activate);

    Layer3.pre_activate = dot(Layer2.activated, Layer3.W, Layer2.W_shape[0], Layer2.W_shape[1], Layer3.W_shape[1]);
    Layer3.activated = Layer3.f(Layer3.pre_activate);

    vector<float> predict = Layer3.activated;

    return predict;
}

void Neural_Network::backprop(vector<float> y, vector<float> pred, int batch_size) {
    vector<float> pred_error = y - pred;
    Layer3.W_delta = pred_error * Layer3.d_f(pred);
    Layer2.W_delta = dot(Layer3.W_delta, transpoose(Layer3.W, Layer3.W_shape[0], Layer3.W_shape[1]), batch_size, Layer3.W_shape[1], Layer3.W_shape[0]) * Layer2.d_f(Layer2.activated);
    Layer1.W_delta = dot(Layer2.W_delta, transpoose(Layer2.W, Layer2.W_shape[0], Layer2.W_shape[1]), batch_size, Layer2.W_shape[1], Layer2.W_shape[0]) * Layer1.d_f(Layer1.activated);
    Layer3.dE_dW = dot(transpoose(Layer2.activated, batch_size, Layer3.W_shape[0]), Layer3.W_delta, Layer3.W_shape[0], batch_size, Layer3.W_shape[1]);
    Layer2.dE_dW = dot(transpoose(Layer1.activated, batch_size, Layer2.W_shape[0]), Layer2.W_delta, Layer2.W_shape[0], batch_size, Layer2.W_shape[1]);
    Layer1.dE_dW = dot(transpoose(Layer0.activated, batch_size, Layer1.W_shape[0]), Layer1.W_delta, Layer1.W_shape[0], batch_size, Layer1.W_shape[1]);
    Layer3.W = Layer3.W + Layer3.dE_dW;
    Layer2.W = Layer2.W + Layer2.dE_dW;
    Layer1.W = Layer1.W + Layer1.dE_dW;
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
