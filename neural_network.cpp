//
//  10lines_neural_network.cpp
//
// clang++ -std=c++11 neural_network.cpp
//
//

#include <iostream>
#include"Eigen/Core"
#include "neural_network.h"

using namespace std;
using Eigen::MatrixXf;


//
// プロトタイプ宣言
//
// void print(const vector<float>&, int, int);



int main(int argc, const char * argv[]) {

    Eigen::MatrixXf data(4,4);
    data << 5.1, 3.5, 1.4, 0.2,	//インスタンス1
            4.9, 3.0, 1.4, 0.2,	//インスタンス2
            6.2, 3.4, 5.4, 2.3, //インスタンス3
            5.9, 3.0, 5.1, 1.8;	//インスタンス4
        //    5.8, 2.8, 5.0, 2.0; //インスタンス5

    MatrixXf label(4,2);
    label << 1, 0,
             1, 0,
             0, 1,
        //     0, 1,
             0, 1;

    MatrixXf W1_(4, 6);
    W1_ <<  0.5,  0.5,  0.5, -0.5,  0.1,  0.5,
            0.5,  0.1, -0.5,  0.5,  0.5,  0.1,
            0.5, -0.5,  0.1,  0.5,  0.5,  0.5,
           -0.5,  0.5,  0.5,  0.1,  0.5,  0.5;

    MatrixXf b1_(1, 6);
    b1_ <<  0.0,  0.0,  0.0,  0.0,  0.0,  0.0;

    MatrixXf W2_(6, 3);
    W2_ <<  0.1,  0.5, -0.5,
            0.5, -0.1,  0.5,
           -0.5,  0.5,  0.1,
            0.1,  0.5, -0.5,
            0.5, -0.1,  0.5,
           -0.5,  0.5,  0.1;

    MatrixXf b2_(1, 3);
    b2_ <<  0.0,  0.0,  0.0;

    MatrixXf W3_(3, 2);
    W3_ <<  0.1,  0.5,
            0.5,  0.1,
            0.1,  0.5;

    MatrixXf b3_(1, 2);
    b3_ <<  0.0,  0.0;


    Neural_Network nn;
    nn.build_fullConnectedLayer(W1_, 4, 6,
                               b1_, 6, false,
                               sigmoid, sigmoid_d);
    nn.build_fullConnectedLayer(W2_, 6, 3,
                               b2_, 3, false,
                               sigmoid, sigmoid_d);
    nn.build_fullConnectedLayer(W3_, 3, 2,
                               b3_, 2, false,
                               sigmoid, sigmoid_d);
    nn.allocateMemory(4);
    MatrixXf pred;
    int epoch = 1000;

	for (unsigned int i = 0; i != epoch; ++i) {
        pred = nn.forwardprop(data);
        nn.backprop(label, pred);

		if ( i ==  epoch-1 ) {
		 	cout << pred << endl;
		}
	}
	return 0;
}
