#ifndef INCLUDE_mnist_h_
#define INCLUDE_mnist_h_

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <unistd.h>
#include "Eigen/Core"
#include "batch.h"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using Eigen::MatrixXf;

static const string MNIST_DATA_DIR("data_set/mnist");
static const unsigned int INITIAL_TRAIN_SAMPE_SEED = 123;
static const unsigned int INITIAL_TEST_SAMPE_SEED = 456;
static const unsigned int INITIAL_VALID_SAMPE_SEED = 789;



class Mnist {
public:
    MatrixXf read_exampleFile(string data_path);
    MatrixXf read_labelFile(string data_path);
    Batch _train = Batch(MatrixXf::Zero(1, 1), MatrixXf::Zero(1, 1));
    Batch _test = Batch(MatrixXf::Zero(1, 1), MatrixXf::Zero(1, 1));
    Batch _valid = Batch(MatrixXf::Zero(1, 1), MatrixXf::Zero(1, 1));

    Mnist(const unsigned int train_num=60000, const unsigned int train_init_seed=INITIAL_TRAIN_SAMPE_SEED,
          const unsigned int test_num=10000, const unsigned int test_init_seed=INITIAL_TEST_SAMPE_SEED,
          const unsigned int valid_num=0, const unsigned int valid_init_seed=INITIAL_VALID_SAMPE_SEED,
          const string data_dir=MNIST_DATA_DIR);
private:
    int _reverseInt(int i);
    string data_dir = MNIST_DATA_DIR;
};


Mnist::Mnist(const unsigned int train_num, const unsigned int train_init_seed,
             const unsigned int test_num, const unsigned int test_init_seed,
             const unsigned int valid_num, const unsigned int valid_init_seed,
             const string data_dir) {
    this->data_dir = data_dir;

    // Read sample files.
    char work_dir[1023];
    getcwd(work_dir,1023);
    MatrixXf train_example = read_exampleFile(string(work_dir)+"/"+data_dir+"/train-images-idx3-ubyte");
    MatrixXf train_label = read_labelFile(string(work_dir)+"/"+data_dir+"/train-labels-idx1-ubyte");
    MatrixXf test_example = read_exampleFile(string(work_dir)+"/"+data_dir+"/t10k-images-idx3-ubyte");
    MatrixXf test_label = read_labelFile(string(work_dir)+"/"+data_dir+"/t10k-labels-idx1-ubyte");

    this->_train = Batch(train_example/255.f, train_label);
    this->_test = Batch(test_example/255.f, test_label);
    this->_valid = this->_train.split(valid_num, valid_init_seed);
}


int Mnist::_reverseInt(int i) {
    /*
        Convert byte string to int
    */
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


MatrixXf Mnist::read_exampleFile(string file_path) {
    ifstream fin(file_path, std::ios::in | std::ios::binary);
    if(!fin){
        cout << file_path << endl;
        cout << "ファイルパスが間違っています。" << endl;
        exit(1);
     }

    int magic_number = 0;
    int sample_size = 0;
    int rows = 0;
    int cols = 0;

    // Read file header.
    fin.read((char*)&magic_number, sizeof(magic_number));
    magic_number = this->_reverseInt(magic_number);
    fin.read((char*)&sample_size,sizeof(sample_size));
	sample_size = this->_reverseInt(sample_size);
	fin.read((char*)&rows,sizeof(rows));
	rows = this->_reverseInt(rows);
	fin.read((char*)&cols,sizeof(cols));
	cols = this->_reverseInt(cols);
    cout << "read mnist_training_example_file  " << magic_number << " " << sample_size <<  " " << rows << " " << cols << endl;

    MatrixXf example(sample_size, rows*cols);

    for ( int i = 0; i < sample_size; i++ ) {
        for ( int row = 0; row < rows; row++ ) {
            for ( int col = 0; col < cols; col++ ) {
                unsigned char temp = 0;
                fin.read((char*)&temp, sizeof(temp));
                example(i, rows*row+col) = (float)temp;
            }
        }
    }

    return example;
}


MatrixXf Mnist::read_labelFile(string file_path) {
    ifstream fin(file_path, std::ios::in | std::ios::binary);
    if(!fin){
        cout << file_path << endl;
        cout << "ファイルパスが間違っています。" << endl;
        exit(1);
     }

    int magic_number = 0;
	int sample_size = 0;

	// Read file header.
	fin.read((char*)&magic_number,sizeof(magic_number));
	magic_number = this->_reverseInt(magic_number);
	fin.read((char*)&sample_size,sizeof(sample_size));
	sample_size = this->_reverseInt(sample_size);
    cout << "read mnist_training_label_file  " << magic_number << " " << sample_size << endl;

    MatrixXf label = MatrixXf::Zero(sample_size, 10);

    for ( int i = 0; i < sample_size; i++ ) {
        unsigned char temp = 0;
        fin.read((char*)&temp, sizeof(temp));
        label(i, (int)temp) = 1.f;
    }

    return label;
}


#endif // INCLUDE_mnist_h_
