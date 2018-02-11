#include <vector>
#include <math.h>
#include "my_math.h"

using std::vector;

//
// training data
//
vector<float> X {
    5.1, 3.5, 1.4, 0.2,	//インスタンス1
    4.9, 3.0, 1.4, 0.2,	//インスタンス2
    6.2, 3.4, 5.4, 2.3, //インスタンス3
    5.9, 3.0, 5.1, 1.8	//インスタンス4
};

vector<float> y {
    1, 0,
    1, 0,
    0, 1,
    0, 1
};

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


class Neural_Network {
    /*
    */
public:
    vector<float> forwardprop(void);
    void backprop(vector<float>);

private:
    vector<float> L1;
    vector<float> L2;
};

vector<float> Neural_Network::forwardprop(void) {
    L1 = sigmoid(dot(X, W1, 4, 4, 6));
    L2 = sigmoid(dot(L1, W2, 4, 6, 3));
    vector<float> pred = sigmoid(dot(L2, W3, 4, 3, 2));

    return pred;
}

void Neural_Network::backprop(vector<float> pred) {
    vector<float> pred_error = y - pred;
    vector<float> W3_delta = pred_error * sigmoid_d(pred);
    vector<float> W2_delta = dot(W3_delta, transpoose(W3, 3, 2), 4, 2, 3) * sigmoid_d(L2);
    vector<float> W1_delta = dot(W2_delta, transpoose(W2, 6, 3), 4, 3, 6) * sigmoid_d(L1);
    vector<float> dE_dW3 = dot(transpoose(L2, 4, 3), W3_delta, 3, 4, 2);
    vector<float> dE_dW2 = dot(transpoose(L1, 4, 6), W2_delta, 6, 4, 3);
    vector<float> dE_dW1 = dot(transpoose(X, 4, 4), W1_delta, 4, 4, 6);
    W3 = W3 + dE_dW3;
    W2 = W2 + dE_dW2;
    W1 = W1 + dE_dW1;
}
