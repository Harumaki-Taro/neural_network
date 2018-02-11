//
//  10lines_neural_network.cpp
//
// clang++ -std=c++11  10lines_neural_network.cpp
//
//

#include <iostream>
#include <vector>
#include <math.h>

using std::vector;
using std::cout;
using std::endl;


//
// プロトタイプ宣言
//
vector<float> dot(const vector<float>&, const vector<float>&, const int, const int, const int);
vector<float> sigmoid(const vector<float>&);
vector<float> sigmoid_d(const vector<float>&);
vector<float> operator-(const vector<float>&, const vector<float>&);
vector<float> operator+(const vector<float>&, const vector<float>&);
vector<float> operator*(const vector<float>&, const vector<float>&);
vector<float> transpoose(const vector<float>&, const int, const int);
void print(const vector<float>&, int, int);


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
    0.1, 0.5, 0.5, 0.5, 0.1, 0.5,
    0.5, 0.1, 0.5, 0.5, 0.5, 0.1,
    0.5, 0.5, 0.1, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.1, 0.5, 0.5
};

vector<float> W2 {
	0.1, 0.5, 0.5,
	0.5, 0.1, 0.5,
	0.5, 0.5, 0.1,
	0.1, 0.5, 0.5,
	0.5, 0.1, 0.5,
	0.5, 0.5, 0.1
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



int main(int argc, const char * argv[]) {

    Neural_Network nn;
    vector<float> pred;

	for (unsigned int i = 0; i != 1000; ++i) {
        pred = nn.forwardprop();
        nn.backprop(pred);

		if ( i ==  999) {
			print(pred, 4, 2);
		}
	}
	return 0;
}


vector<float> dot(const vector<float>& m1, const vector<float>& m2,
					const int m1_rows, const int m1_columns, const int m2_columns) {
	/* 	Return the product of two matrices: m1 * m2
		<Inputs>
			m1: vector, left matrix of size m1_rows * m1_columns
			m2: vector, right matrix of size m1_columns * m2_columns
		<Output>
			vector, m1 * m2, product of two vectors m1 and m2
	*/

	vector<float> output (m1_rows*m2_columns);

	for ( int row = 0; row < m1_rows; ++row ) {
		for ( int col = 0; col < m2_columns; ++col ) {
			output[row * m2_columns + col] = 0.f;
			for ( int k = 0; k < m1_columns; k++ ) {
				output[row * m2_columns + col]
					+= m1[row * m1_columns + k] * m2[k * m2_columns + col];
			}
		}
	}

	return output;
}


vector<float> sigmoid(const vector<float>& m1) {
	/*
		Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
		<Inputs>
			m1: vector
		<Output>
			1/(1 + e^-x) for every element of the input matrix m1.
	 */
	const unsigned long VECTOR_SIZE = m1.size();
	vector <float> output(VECTOR_SIZE);

	for ( unsigned int i = 0; i < VECTOR_SIZE; i++ ) {
		output[i] = 1 / (1 + exp(-m1[i]));
	}

	return output;
}


vector<float> sigmoid_d(const vector<float>& m1) {
	/*
		Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
		where f(x) is sigmoid function.
		<Inputs>
			m1: vector
		<Output>
			x(1 - x) for every element of the input matrix m1.
	 */
	const unsigned long VECTOR_SIZE = m1.size();
	vector<float> output(VECTOR_SIZE);

	for ( unsigned i = 0; i < VECTOR_SIZE; ++i ) {
		output[i] = m1[i] * (1 - m1[i]);
	}

	return output;
}


vector<float> operator-(const vector<float>& m1, const vector<float>& m2) {
	/*
		Returns the difference between two vectors.
		<Inputs>
			m1: vector
			m2: vector
		<Output>
			vector, m1 - m2
	 */

	const unsigned long VECTOR_SIZE = m1.size();
	vector<float> difference(VECTOR_SIZE);

	for ( unsigned i = 0; i < VECTOR_SIZE; i++) {
		difference[i] = m1[i] - m2[i];
	}

	return difference;
}


vector<float> operator+(const vector<float>& m1, const vector<float>& m2) {
	/*
		Returns the elementwise sum of two vectors.
		<Inputs>
			m1: vector
			m2: vector
		<Output>
			vector, m1 + m2
	 */
	const unsigned long VECTOR_SIZE = m2.size();
	vector<float> sum(VECTOR_SIZE);

	for ( unsigned i = 0; i < VECTOR_SIZE; i++ ) {
		sum[i] = m1[i] + m2[i];
	}

	return sum;
}


vector<float> operator*(const vector<float>& m1, const vector<float>& m2) {
	/*
		Returns the product of two vectors (elementwise multiplication).
		<Inputs>
			m1: vector
			m2: vector
		<Output>
			vector, m1 * m2, elementwise multiplication
	 */

	const unsigned long VECTOR_SIZE = m1.size();
	vector<float> product(VECTOR_SIZE);

	for ( unsigned i = 0; i < VECTOR_SIZE; i++ ) {
		product[i] = m1[i] * m2[i];
	}

	return product;
}


vector<float> transpoose(const vector<float>& m, const int C, const int R) {
	/*
		Returns a transposed matrix of input matrix.
		<Inputs>
			m: vector, input matrix
			C: int, number of columns in the input matrix
			R: int, number of rows in the input matrix
		<Output>
			vector, transposed matrix mT of the input matrix m
	*/

	vector<float> mT(C*R);

	for ( int n = 0; n < C*R; n++ ) {
		int i = n / C;
		int j = n % C;
		mT[n] = m[R*j + i];
	}

	return mT;
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
