#include <iostream>
#include <math.h>
#include "Eigen/Core"
#include "Eigen/Geometry"

using std::cout;
using std::endl;
using Eigen::MatrixXf;
using Eigen::ArrayXXf;


MatrixXf sigmoid(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
		<Inputs>
			m1: vector
		<Output>
			1/(1 + e^-x) for every element of the input matrix m1.
	 */
    return (1.f / (1.f + (-m.array()).exp())).matrix();
}


MatrixXf sigmoid_d(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
		where f(x) is sigmoid function.
		<Inputs>
			m1: vector
		<Output>
			x(1 - x) for every element of the input matrix m1.
	 */
     return (m.array() * (1.f - m.array())).matrix();
}


MatrixXf sum(const MatrixXf m, const int axis) {
    /*
    */
    if ( axis == 0 ) {
        return (m.colwise().sum()).matrix();
    } else if ( axis == 1 ) {
        return (m.rowwise().sum()).matrix();
    } else {
        cout << "sumのaxis指定が間違っています.";
        exit(1);
    }
}


MatrixXf softmax(const MatrixXf m) {
    /*
    */
    ArrayXXf tmp = m.array();
    // prevent overflow
    tmp -= m.maxCoeff();
    
    return (tmp.exp() / tmp.rowwise().sum().array().exp()).matrix();
}


MatrixXf elemntwiseProduct(const MatrixXf m1, const MatrixXf m2) {
    return (m1.array() * m2.array()).matrix();
}


MatrixXf uniform_rand(const int (&shape)[2], const float max, const float min) {
    MatrixXf output = MatrixXf::Random(shape[0], shape[1]);
    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}

MatrixXf uniform_rand(const int shape, const float max, const float min) {
    MatrixXf output = MatrixXf::Random(1, shape);
    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}
