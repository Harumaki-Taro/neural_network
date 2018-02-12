#include <iostream>
#include <math.h>
#include"Eigen/Core"
#include"Eigen/Geometry"

using std::cout;
using std::endl;
using Eigen::MatrixXf;


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


MatrixXf elemntwiseProduct(const MatrixXf m1, const MatrixXf m2) {
    return (m1.array() * m2.array()).matrix();
}
