#ifndef INCLUDE_dropout_h_
#define INCLUDE_dropout_h_

#include <iostream>
#include <functional>
#include <math.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
using std::shared_ptr;

// NOTE:全結合もチャンネル化した方がいい。
// NOTE:テスト時とトレーニング時で挙動を変える。

class Dropout : public Layer {
public:
    virtual void forwardprop(const vector<vector <MatrixXf> > X);
    virtual void calc_delta(const std::shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const std::shared_ptr<Layer> &prev_layer);

    vector<int> extract_dropoutIndex(void);

    Dropout(const int channel_num, const float dropout_rate=0.5, const int seed=0);

    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual int get_batch_size(void);
    virtual int get_channel_num(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);

private:
    bool trainable = false;
    string type = "dropout";
    bool is_tensor;
    int channel_num;
    float dropout_rate;
    int seed;
    int unit_num;
    int dropout_num;
    int batch_size;
    vector<int> X_shape;
    vector<int> delta_shape;
    vector<int> dropout_index;
};


Dropout::Dropout(const int channel_num, const float dropout_rate, const int seed) {
    this->channel_num = channel_num;
    this->dropout_rate = dropout_rate;

    this->seed = pop_seed(seed);
}


void Dropout::forwardprop(const vector<vector<MatrixXf> > X) {
    this->dropout_index = extract_dropoutIndex();
    for ( int i = 0; i < this->X_shape[0]; ++i ) {
        for ( int j = 0; j < this->X_shape[1]; ++j ) {
            this->_activated[i][j] = X[i][j];
        }
    }

    int m = X_shape[1] * X_shape[2] * X_shape[3];
    int n = X_shape[2] * X_shape[3];
    int o = X_shape[3];
    for ( int i = 0; i < dropout_index.size(); ++i ) {
        int p = this->dropout_index[i] / m;
        int q = (this->dropout_index[i] % m) / n;
        int r = ((this->dropout_index[i] % m) % n) / o;
        int s = ((this->dropout_index[i] % m) % n) % o;
        this->_activated[p][q](r, s) = 0.f;
    }
}


void Dropout::calc_delta(const std::shared_ptr<Layer> &next_layer) {
    for ( int i = 0; i < this->X_shape[0]; ++i ) {
        for ( int j = 0; j < this->X_shape[1]; ++j ) {
            this->delta[i][j] = next_layer->delta[i][j];
        }
    }

    int m = X_shape[1] * X_shape[2] * X_shape[3];
    int n = X_shape[2] * X_shape[3];
    int o = X_shape[3];
    for ( int i = 0; i < this->dropout_index.size(); ++i ) {
        int p = this->dropout_index[i] / m;
        int q = (this->dropout_index[i] % m) / n;
        int r = ((this->dropout_index[i] % m) % n) / o;
        int s = ((this->dropout_index[i] % m) % n) % o;
        this->delta[p][q](r, s) = 0.f;
    }
}


vector<int> Dropout::extract_dropoutIndex(void) {
    std::mt19937 mt(this->seed);
    std::uniform_int_distribution<int> gen_rand(0, INT_MAX);

    std::vector<int> indexes(this->unit_num);
    for ( int i = 0; i < this->unit_num; ++i ) {
        indexes[i] = i;
    }

    for ( int i = 0; i < this->dropout_num; i++ ) {
        int pos = gen_rand(mt) % (this->unit_num - i);
        int tmp = indexes[i];
        indexes[i] = indexes[pos+i];
        indexes[pos+i] = tmp;
    }
    indexes.erase(indexes.end()-this->dropout_num+1, indexes.end());
    indexes.shrink_to_fit(); // NOTE:アクセスできちゃった

    this->seed += 7;

    return indexes;
}


void Dropout::allocate_memory(const int batch_size, const std::shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->unit_num = prev_layer->get_unit_num();
    this->dropout_num = (int)floor((float)this->unit_num * this->dropout_rate);

    this->X_shape.resize(4);
    this->X_shape[0] = prev_layer->_activated.size();
    this->X_shape[1] = prev_layer->_activated[0].size();
    this->X_shape[2] = prev_layer->_activated[0][0].rows();
    this->X_shape[3] = prev_layer->_activated[0][0].cols();

    // activated
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->delta.push_back(tmp_delta);
    }
}


bool Dropout::get_trainable(void) { return this->trainable; }
string Dropout::get_type(void) { return this->type; }
bool Dropout::get_is_tensor(void) { return this->is_tensor; }
int Dropout::get_unit_num(void) { return this->unit_num; }
int Dropout::get_batch_size(void) { return this->batch_size; }
int Dropout::get_channel_num(void) { return this->channel_num; }
vector< vector<MatrixXf> > Dropout::get_activated(void) { return this->_activated; }
vector<vector <MatrixXf> > Dropout::get_delta(void) { return this->delta; }


#endif // INCLUDE_dropout_h_
