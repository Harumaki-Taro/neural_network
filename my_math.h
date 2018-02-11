#include <math.h>
#include <vector>

using std::vector;



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