//
//  10lines_neural_network.cpp
//
// clang++ -std=c++11 neural_network.cpp
//
//

#include <iostream>
#include <vector>
#include <math.h>
#include "neural_network.h"

using std::vector;
using std::cout;
using std::endl;


vector<float> data {
    5.1, 3.5, 1.4, 0.2,	//インスタンス1
    4.9, 3.0, 1.4, 0.2,	//インスタンス2
    6.2, 3.4, 5.4, 2.3, //インスタンス3
    5.9, 3.0, 5.1, 1.8	//インスタンス4
};

vector<float> label {
    1, 0,
    1, 0,
    0, 1,
    0, 1
};


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


//
// プロトタイプ宣言
//
void print(const vector<float>&, int, int);



int main(int argc, const char * argv[]) {

    Neural_Network nn;
    nn.buildFullConnectedLayer(W1_, 4, 6, sigmoid, sigmoid_d);
    nn.buildFullConnectedLayer(W2_, 6, 3, sigmoid, sigmoid_d);
    nn.buildFullConnectedLayer(W3_, 3, 2, sigmoid, sigmoid_d);
    vector<float> pred;
    int epoch = 1000;

	for (unsigned int i = 0; i != epoch; ++i) {
        pred = nn.forwardprop(data, 4, 4);
        nn.backprop(label, pred, 4);

		if ( i ==  epoch-1) {
			print(pred, 4, 2);
		}
	}
	return 0;
}


void print(const vector<float>& m, int n_rows, int n_columns) {
	/*
		"Couts" the input vector as n_rows * n_columns matrix.
		<Inputs>
			m: vector, matrix of size n_rows * n_columns
			n_rows: int, number of rows in the matrix m
			n_columns: int, number of columns in the left matrix m
	 */

	for ( int i = 0; i < n_rows; i++ ) {
		for (int j = 0; j < n_columns; ++j ) {
			cout << m[i * n_columns + j] << " ";
		}
		cout << '\n';
	}
	cout << endl;
}
