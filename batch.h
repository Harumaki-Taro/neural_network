#ifndef INCLUDE_batch_h_
#define INCLUDE_batch_h_

#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include "Eigen/Core"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using Eigen::MatrixXf;


typedef struct Mini_Batch{
    MatrixXf example;
    MatrixXf label;
}Mini_Batch;


class Batch {
public:
    string name;
    Mini_Batch pop(const unsigned int mini_batch_size);
    Mini_Batch randomPop(const unsigned int mini_batch_size);
    Batch split(const unsigned int batch_size, const unsigned int init_seed=0);
    void reduce_size(const unsigned int batch_size);
    Batch(const MatrixXf example,
          const MatrixXf label,
          const unsigned int init_seed=0);
    MatrixXf _example;
    MatrixXf _label;
private:
    unsigned int _batch_size;
    // MatrixXf _example;
    unsigned int _example_size;
    // MatrixXf _label;
    unsigned int _label_size;
    unsigned int _itr = 0;
    unsigned int seed;
    void shuffle_dataset(const unsigned int seed);
};


Batch::Batch(const MatrixXf example, const MatrixXf label, const unsigned int init_seed) {
    this->_example = example;
    this->_example_size = this->_example.cols();
    this->_label = label;
    this->_label_size = this->_label.cols();

    if ( this->_example.rows() != this->_label.rows() ) {
        cout << "exampleの数とラベルの数が一致していません。" << endl;
        exit(1);
    }
    this->_batch_size = this->_example.rows();

    if ( init_seed == 0 ) {
        std::random_device rd;
        this->seed = rd();
    } else {
        this->seed = init_seed;
    }

    this->shuffle_dataset(this->seed);
}


Mini_Batch Batch::randomPop(const unsigned int mini_batch_size) {
    /*
        <NOTE>
            端数は無視されます
    */
    if ( this->_itr + mini_batch_size > this->_batch_size ) {
        // shuffle dataset
        this->seed += 10;
        this->shuffle_dataset(seed);
        cout << "shuffle" << endl;
    }

    return this->pop(mini_batch_size);
}


Mini_Batch Batch::pop(const unsigned int mini_batch_size) {
    /*
        <NOTE>
            端数は無視されます
    */
    if ( this->_itr + mini_batch_size > this->_batch_size ) {
        this->_itr = 0;
    }

    // dataset.itr += this->mini_batch_size;
    Mini_Batch output_mini_batch = Mini_Batch{
                                this->_example.block(this->_itr,0,mini_batch_size,this->_example_size),
                                this->_label.block(this->_itr,0,mini_batch_size,this->_label_size)};

    this->_itr += mini_batch_size;

    return output_mini_batch;
}


void Batch::shuffle_dataset(const unsigned int seed) {
    // 置換行列を作成（各列と各行に唯一つの非ゼロ成分を持つ行列）
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm_row(this->_batch_size);

    // 対角要素に1をセット
    perm_row.setIdentity();

    // 置換行列をシャッフル
    std::shuffle(perm_row.indices().data(), perm_row.indices().data() + perm_row.indices().size(), std::mt19937(seed));

    // 左からかける
    this->_example = perm_row * this->_example;
    this->_label = perm_row * this->_label;
}


Batch Batch::split(const unsigned int new_batch_size, const unsigned int init_seed) {
    if ( new_batch_size > this->_batch_size ) {
        cout << "元のサンプルサイズよりnew_batch_sizeの方が大きいです" << endl;
        exit(1);
    }

    MatrixXf new_example = this->_example.block(this->_batch_size-new_batch_size,0,new_batch_size, this->_example_size);
    MatrixXf new_label = this->_label.block(this->_batch_size-new_batch_size,0,new_batch_size, this->_label_size);

    this->reduce_size(this->_batch_size - new_batch_size);

    return Batch(new_example, new_label, init_seed);
}


void Batch::reduce_size(const unsigned int new_batch_size) {
    if ( new_batch_size > this->_batch_size ) {
        cout << "元のサンプルサイズよりnew_batch_sizeの方が大きいです" << endl;
        exit(1);
    }
    this->_batch_size = new_batch_size;
    this->_example = this->_example.block(0,0,this->_batch_size, this->_example_size);
    this->_label = this->_label.block(0,0,this->_batch_size, this->_label_size);
}


#endif // INCLUDE_batch_h_
