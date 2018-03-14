#ifndef INCLUDE_my_math_h_
#define INCLUDE_my_math_h_

#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using Eigen::MatrixXf;
using Eigen::ArrayXXf;
using Eigen::pow;
using Eigen::log;



MatrixXf sigmoid(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
		<Inputs>
			m: matrix
		<Output>
			1/(1 + e^-x) for every element of the input matrix m.
	*/
    return (1.f / (1.f + (-m.array()).exp())).matrix();
}


MatrixXf sigmoid_d(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
		where f(x) is sigmoid function.
		<Inputs>
			m: matrix
		<Output>
			x(1 - x) for every element of the input matrix m.
	*/
    return (m.array() * (1.f - m.array())).matrix();
}


MatrixXf identity(const MatrixXf m) {
    /*
        Returns the value of the identity function f(x) = x.
        <Input>
            m: matrix
        <Output>
            x for every element of the input matrix m.
    */
    return m;
}


MatrixXf identity_d(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function derivative f'(x) = 1,
		where f(x) is sigmoid function.
		<Inputs>
			m: matrix
		<Output>
			1 for every element of the input matrix m.
	*/
    return MatrixXf::Ones(m.rows(), m.cols());
}


MatrixXf tanh_(const MatrixXf m) {
    /*
        Returns the value of the identity function f(x) = tanh(x).
        <Input>
            m: matrix
        <Output>
            relu(x) for every element of the input matrix m.
    */
    return m.array().tanh().matrix();
}


MatrixXf tanh_d(const MatrixXf m) {
    /*
        Returns the value of the tanh function f'(x) = 1 - f(x)^2.
        <Input>
            m: matrix
        <Output>
            1 - x^2 for every element of the input matrix m
    */
    return (1.f - pow(m.array(), 2)).matrix();
}


MatrixXf diff(const MatrixXf m1, const MatrixXf m2) {
    /*
        Returns the value of the tanh function m1 - m1.
        <Input>
            m1: matrix
            m2: matrix
        <Output>
            m1 - m2 for every element of the input matrix m1, m2.
    */
    return (m1.array() - m2.array()).matrix();
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
    MatrixXf output(m.rows(), m.cols());
    ArrayXXf exp_m = m.array();
    // prevent overflow
    exp_m = (exp_m - m.maxCoeff()).exp();
    MatrixXf row_sum = exp_m.rowwise().sum();

    for ( int i = 0; i != m.rows(); i++ ) {
        output.block(i,0,1,m.cols()) = (exp_m.row(i) / row_sum(i)).matrix();
    }

    return output;
}


float mean_square_error(const MatrixXf y, const MatrixXf t) {
    /*
        Returns the value of the mean square error between prediction y and teacher t.
        <Inputs>
            y: MatrixXf, prediction
            t: MatrixXf, teacher
        <Output>
            1/2 * (y - t)^2 for all element of the input matrix.
    */
    return 0.5f * pow(y.array() - t.array(), 2).sum();
}


float mean_cross_entropy(const MatrixXf y, const MatrixXf t) {
    /*
        Returns the value of the cross entropy error between prediction y and teacher t.
        <Inputs>
            y: MatrixXf, prediction
            t: MatrixXf, teacher
        <Output>
            - sum(t * log(y)) for all element of the input matrix.
        <Note>
            - Add 1e-7 to prevent log from divergence.
    */
    float batch_size = (float)t.rows();

    return - (t.array() * (y.array()+1e-7).log()).sum() / batch_size;
}


float mean_cross_entropy_one_of_k(const MatrixXf y, const MatrixXf t) {
    /*
        Returns the value of the cross entropy error between prediction y and teacher t.
        <Inputs>
            y: MatrixXf, prediction
            t: MatrixXf, teacher
        <Output>
            - sum(t * log(y)) for all element of the input matrix.
        <Note>
            - Add 1e-7 to prevent log from divergence.
    */
    float batch_size = (float)t.rows();

    MatrixXf output(y.rows(), 1);
    MatrixXf::Index max_index;
    for ( int i = 0; i < t.rows(); i++ ) {
        t.row(i).maxCoeff(&max_index);
        output(i,0) = log(y(i,max_index)+1e-7);
    }

    return - output.sum() / batch_size;
}


MatrixXf elemntwiseProduct(const MatrixXf m1, const MatrixXf m2) {
    return (m1.array() * m2.array()).matrix();
}


vector<vector <MatrixXf> > uniform_rand(const int (&shape)[4], const float min, const float max,
                                        const int seed=0) {
    // set random seed
    int _seed;
    if ( max - min <= 0.f ) {
        cout << "minよりmaxの方が小さいです" << endl;
        exit(1);
    }
    if ( seed == 0 ) {
        std::random_device rnd;
        _seed = rnd();
    } else {
        _seed = seed;
    }

    // Mel sense twister
    std::mt19937 mt(_seed);
    std::uniform_int_distribution<float> gen_rand(min, max);

    // allocate memory
    vector <vector <MatrixXf> > output;
    output.resize(shape[0]);
    for ( int i = 0; i < shape[0]; i++ ) {
        output[i].resize(shape[1]);
        for ( int j = 0; j < shape[1]; j++ ) {
            output[i][j].resize(shape[2], shape[3]);
        }
    }

    // set random value
    for ( int i = 0; i < shape[0]; i++ ) {
        for ( int j = 0; j < shape[1]; j++ ) {
            for ( int k = 0; k < shape[2]; k++ ) {
                for ( int l = 0; l < shape[3]; l++ ) {
                    output[i][j](k,l) = gen_rand(mt);
                }
            }
        }
    }

    return output;
}


MatrixXf uniform_rand(const int (&shape)[2], const float min, const float max) {
    if ( max - min <= 0.f ) {
        cout << "minよりmaxの方が小さいです" << endl;
        exit(1);
    }
    MatrixXf output = MatrixXf::Random(shape[0], shape[1]);
    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}


MatrixXf uniform_rand(const int shape, const float min, const float max) {
    if ( max - min <= 0.f ) {
        cout << "minよりmaxの方が小さいです" << endl;
        exit(1);
    }
    MatrixXf output = MatrixXf::Random(1, shape);
    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}

#endif // INCLUDE_my_math_h_
