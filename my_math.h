#ifndef INCLUDE_my_math_h_
#define INCLUDE_my_math_h_

#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include <limits>
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

static const float PI = 3.14159265359;


void check_nan(MatrixXf m, string comment) {
    for ( int i = 0; i < m.rows(); ++i ) {
        for ( int j = 0; j < m.cols(); ++j ) {
            float x = m(i, j);
            if ( isnan(x) ) {
                cout << "値がnanです。 " << comment << "で発生しました。" << endl;
                exit(1);
            } else if ( isinf(x) ) {
                cout << "出力がinfです。" << comment << "で発生しました。" << endl;
                exit(1);
            }
        }
    }
}


void check_nan(float x, string comment) {
    if ( isnan(x) ) {
        cout << "値がnanです。 " << comment << "で発生しました。" << endl;
        exit(1);
    } else if ( isinf(x) ) {
        cout << "出力がinfです。" << comment << "で発生しました。" << endl;
        exit(1);
    }
}


MatrixXf sigmoid(const MatrixXf m) {
	/*
		Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
		<Inputs>
			m: matrix
		<Output>
			1/(1 + e^-x) for every element of the input matrix m.
	*/
    MatrixXf output = (1.f / (1.f + (-m.array()).exp())).matrix();

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
    MatrixXf output = (m.array() * (1.f - m.array())).matrix();

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
}



MatrixXf relu(const MatrixXf m) {
    /*
        Returns the value of the relu function f(x) = max(0, x).
        <Inputs>
            m: matrix
        <Output>
            max(0, x) for every element of the input matrix m.
    */
    MatrixXf output = MatrixXf::Zero(m.rows(), m.cols());

    for ( int i = 0; i < m.rows(); ++i ) {
        for ( int j = 0; j < m.cols(); ++j ) {
            if ( m(i, j) > 0.f ) {
                output(i, j) = m(i, j);
            }
        }
    }

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
}


MatrixXf relu_d(const MatrixXf m) {
    /*
        Returns the value of the relu function derivative f'(x) = 1 if x>0 else 0,
        where f(x) is sigmoid function.
        <Inputs>
            m: matrix
        <Output>
            1 if x>0 else 0 for every element of the input matrix m.
    */
    MatrixXf output = MatrixXf::Zero(m.rows(), m.cols());

    for ( int i = 0; i < m.rows(); ++i ) {
        for ( int j = 0; j < m.cols(); ++j ) {
            if ( m(i, j) >= 0.f ) {
                output(i, j) = 1.f;
            }
        }
    }

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
    MatrixXf output = m.array().tanh().matrix();

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
}


MatrixXf tanh_d(const MatrixXf m) {
    /*
        Returns the value of the tanh function f'(x) = 1 - f(x)^2.
        <Input>
            m: matrix
        <Output>
            1 - x^2 for every element of the input matrix m
    */
    MatrixXf output = (1.f - pow(m.array(), 2)).matrix();

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
    MatrixXf output = m1 - m2;

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
    MatrixXf exp_m(m.rows(), m.cols());
    MatrixXf exp_sum(m.rows(), 1);
    for ( int i = 0; i < m.rows(); ++i ) {
        MatrixXf m_col = m.block(i,0,1,m.cols());
        // prevent overflow
        exp_m.block(i,0,1,m.cols()) = (m_col.array() - m_col.maxCoeff()).exp().matrix();
        exp_sum(i,0) = exp_m.block(i,0,1,m.cols()).sum();
    }

    for ( int i = 0; i != m.rows(); i++ ) {
        output.block(i,0,1,m.cols()) = exp_m.row(i) / exp_sum(i, 0);
    }

    check_nan(output, "my_math/mean_cross_entropy");

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
    float output = 0.5f * pow(y.array() - t.array(), 2).sum();

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
            - Add 1e-6 to prevent log from divergence.
    */
    float batch_size = (float)t.rows();
    float output = - (t.array() * (y.array()+1e-7).log()).sum() / batch_size;

    check_nan(output, "my_math/mean_cross_entropy");

    return output;
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
            - Add 1e-6 to prevent log from divergence.
    */
    float batch_size = (float)t.rows();

    MatrixXf output(y.rows(), 1);
    MatrixXf::Index max_index;
    for ( int i = 0; i < t.rows(); i++ ) {
        t.row(i).maxCoeff(&max_index);
        output(i,0) = log(y(i,max_index)+1e-7);
    }
    float _output_ = - output.sum() / batch_size;

    check_nan(_output_, "my_math/mean_cross_entropy_one_of_k");

    return _output_;
}


MatrixXf elemntwiseProduct(const MatrixXf m1, const MatrixXf m2) {
    MatrixXf output = (m1.array() * m2.array()).matrix();

    check_nan(output, "my_math/elementwiseProduct");

    return output;
}


int pop_seed(int seed=0) {
    /*
        Returns non-deterministic random number if seed is 0.
        Otherwise, return the same number.
    */
    int _seed;
    if ( seed == 0 ) {
        std::random_device rnd;
        _seed = rnd();
    } else {
        _seed = seed;
    }

    return _seed;
}


vector<int> rand_array(const int num, const int min=0, const int max=1, const int seed=0) {
    int _seed = pop_seed(seed);
    std::mt19937 mt(_seed);
    std::uniform_int_distribution<int> gen_rand(min, max);

    vector<int> output;
    for ( int i = 0; i < num; i++ ) {
        output.push_back(gen_rand(mt));
    }

    return output;
}


vector<vector <MatrixXf> > uniform_rand(const int (&shape)[4], const float min, const float max,
                                        const int seed=0) {

    if ( max - min <= 0.f ) {
        cout << "minよりmaxの方が小さいです" << endl;
        exit(1);
    }

    // Mel sense twister
    int _seed = pop_seed(seed);
    std::mt19937 mt(_seed);
    std::uniform_real_distribution<float> gen_rand(min, max);

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
            check_nan(output[i][j], "my_math/uniform_rand/4");
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

    check_nan(output, "my_math/uniform_rand/2");

    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}


MatrixXf uniform_rand(const int shape, const float min, const float max) {
    if ( max - min <= 0.f ) {
        cout << "minよりmaxの方が小さいです" << endl;
        exit(1);
    }
    MatrixXf output = MatrixXf::Random(1, shape);

    check_nan(output, "my_math/uniform_rand/1");

    return (((output.array() / 2.f) + 0.5f) * (max - min) + min).matrix();
}


vector< vector<MatrixXf> > gauss_rand(const int (&shape)[4], const float mu, const float sgm,
                                      const int seed=0) {
    /*
        Returns normal random number
    */

    // Mel sense twister
    int _seed = pop_seed(seed);
    std::mt19937 mt(_seed);
    std::normal_distribution<float> gen_rand(mu, sgm);

    // allocate memory
    vector< vector<MatrixXf> > output;

    // set random value
    for ( int i = 0; i < shape[0]; i++ ) {
        vector<MatrixXf> tmp;
        for ( int j = 0; j < shape[1]; j++ ) {
            tmp.push_back(MatrixXf::Zero(shape[2], shape[3]));
            for ( int k = 0; k < shape[2]; k++ ) {
                for ( int l = 0; l < shape[3]; l++ ) {
                    tmp[j](k, l) = gen_rand(mt);
                }
            }
        }
        output.push_back(tmp);
    }

    for ( int i = 0; i < output.size(); ++i ) {
        for ( int j = 0; j < output[0].size(); ++j ) {
            check_nan(output[i][j], "my_math/gauss_rand/4");
        }
    }

    return output;
}


MatrixXf gauss_rand(const int (&shape)[2], const float mu, const float sgm,
                                      const int seed=0) {
    /*
        Returns normal random number
    */

    // Mel sense twister
    int _seed = pop_seed(seed);
    std::mt19937 mt(_seed);
    std::normal_distribution<float> gen_rand(mu, sgm);

    // allocate memory
    MatrixXf output(shape[0], shape[1]);

    // set random value
    for ( int i = 0; i < shape[0]; i++ ) {
        for ( int j = 0; j < shape[1]; j++ ) {
            output(i, j) = gen_rand(mt);
        }
    }

    check_nan(output, "my_math/gauss_rand/2");

    return output;
}


MatrixXf gauss_rand(const int shape, const float mu, const float sgm,
                                      const int seed=0) {
    /*
        Returns normal random number
    */

    // Mel sense twister
    int _seed = pop_seed(seed);
    std::mt19937 mt(_seed);
    std::normal_distribution<float> gen_rand(mu, sgm);

    // allocate memory
    MatrixXf output(1, shape);

    // set random value
    for ( int i = 0; i < shape; i++ ) {
        output(0, i) = gen_rand(mt);
    }

    check_nan(output, "my_math/gauss_rand/1");

    return output;
}


#endif // INCLUDE_my_math_h_
