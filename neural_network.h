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


vector<float> W1 {
     0.1,  0.5,  0.5, -0.5,  0.1,  0.5,
     0.5,  0.1, -0.5,  0.5,  0.5,  0.1,
     0.5, -0.5,  0.1,  0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,  0.1,  0.5,  0.5
};

vector<float> W2 {
	 0.1,  0.5, -0.5,
	 0.5, -0.1,  0.5,
	-0.5,  0.5,  0.1,
	 0.1,  0.5, -0.5,
	 0.5, -0.1,  0.5,
	-0.5,  0.5,  0.1
};


vector<float> W3 {
	0.1, 0.5,
	0.5, 0.1,
	0.1, 0.5
};


typedef struct {
    vector<float> delta;
    vector<float> dE_dW;
}BackUnit;


class Neural_Network {
    /*
    */
public:
    vector<float> forwardprop(vector<float>);
    void backprop(vector<float>, vector<float>);
    // void build_Layer(vector<float>, vector<float>, int, int, int, function<vector<float>>);

private:
    list<vector<float>> forward_list;
    list<vector<float>> backward_list;

    vector<float> L0;
    vector<float> L1;
    vector<float> L2;
};

vector<float> Neural_Network::forwardprop(vector<float> X) {
    L0 = X;
    L1 = sigmoid(dot(L0, W1, 4, 4, 6));
    L2 = sigmoid(dot(L1, W2, 4, 6, 3));
    vector<float> pred = sigmoid(dot(L2, W3, 4, 3, 2));

    return pred;
}

void Neural_Network::backprop(vector<float> y, vector<float> pred) {
    vector<float> pred_error = y - pred;
    vector<float> W3_delta = pred_error * sigmoid_d(pred);
    vector<float> W2_delta = dot(W3_delta, transpoose(W3, 3, 2), 4, 2, 3) * sigmoid_d(L2);
    vector<float> W1_delta = dot(W2_delta, transpoose(W2, 6, 3), 4, 3, 6) * sigmoid_d(L1);
    vector<float> dE_dW3 = dot(transpoose(L2, 4, 3), W3_delta, 3, 4, 2);
    vector<float> dE_dW2 = dot(transpoose(L1, 4, 6), W2_delta, 6, 4, 3);
    vector<float> dE_dW1 = dot(transpoose(L0, 4, 4), W1_delta, 4, 4, 6);
    W3 = W3 + dE_dW3;
    W2 = W2 + dE_dW2;
    W1 = W1 + dE_dW1;
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
