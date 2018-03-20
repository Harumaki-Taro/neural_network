#ifndef INCLUDE_full_connect_layer_h_
#define INCLUDE_full_connect_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
using std::shared_ptr;

class FullConnect_Layer : public Layer {
public:
    virtual void forwardprop(const vector<vector <MatrixXf> > X);
    virtual void calc_delta(const std::shared_ptr<Layer> &next_layer);
    virtual void calc_differential(const std::shared_ptr<Layer> &prev_layer,
                                   const std::shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer);

    FullConnect_Layer(const int channel_num,
                      const int (&W_shape)[2], const string initializer="Xavier",
                      const bool use_bias=true);

    // getter
    virtual bool get_trainable(void);
    virtual int get_batch_size(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual int get_channel_num(void);
    virtual bool get_use_bias(void);
    int get_W_cols(void);
    int get_W_rows(void);
    virtual vector<vector <MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);

    // setter
    virtual void set_batch_size(const int batch_size, const shared_ptr<Layer> &prev_layer);
    virtual string get_type(void);
    // virtual void set_bW(const MatrixXf W, const MatrixXf b, const bool use_bias);
    // virtual void set_W(MatrixXf);
    // virtual void set_b(MatrixXf);
    virtual void set_delta(const vector<vector <MatrixXf> > delta);

private:
    bool trainable = true;
    const string type = "full_connect_layer";
    const bool is_tensor = false;
    int unit_num;
    // Parameters tracked during learning
    // Parameters specified at first
    int batch_size;
    int _W_cols;
    int _W_rows;
    int channel_num;
    bool use_bias;
    // Storage during learning
    vector<vector <MatrixXf> > _input;

    void build_layer(const function<MatrixXf(MatrixXf)> f,
                     const function<MatrixXf(MatrixXf)> d_f,
                     const MatrixXf W, const MatrixXf b,
                     const bool use_bias);
    void build_layer(const function<MatrixXf(MatrixXf)> f,
                     const function<MatrixXf(MatrixXf)> d_f,
                     const int (&W_shape)[2], const string initializer="Xavier",
                     const bool use_bias=true,
                     const float W_min=-0.01, const float W_max=0.01,
                     const float b_min=-0.01, const float b_max=0.01);
};


FullConnect_Layer::FullConnect_Layer(const int channel_num,
                                     const int (&W_shape)[2], const string initializer,
                                     const bool use_bias) {

    this->W.resize(1); this->W[0].resize(channel_num);
    int bW_shape[2] = { W_shape[0]+1, W_shape[1] };
    // Define weight and bias at random.
    for ( int i = 0; i < channel_num; ++i ) {
        if ( initializer == "Xavier" ) {
            this->W[0][i] = gauss_rand(bW_shape, 0.f, sqrt(1.f/(W_shape[0]+1.f)));
        } else if ( initializer == "He" ) {
            this->W[0][i] = gauss_rand(bW_shape, 0.f, sqrt(2.f/(W_shape[0]+1.f)));
        } else {
            cout << "指定の活性化関数に対応していません" << endl;
        }
    }

    this->channel_num = channel_num;
    this->_W_rows = W_shape[0];
    this->_W_cols = W_shape[1];
    this->use_bias = use_bias;
}


void FullConnect_Layer::forwardprop(const vector<vector <MatrixXf> > X) {
    if ( this->channel_num > 1 ) {
        #pragma omp parallel for
        for ( int i = 0; i < this->channel_num; ++i ) {
            this->_input[0][i].block(0,1,this->batch_size,this->_W_rows) = X[0][i];
            this->_activated[0][i] = this->_input[0][i] * this->W[0][i];
        }
    } else {
        this->_input[0][0].block(0,1,this->batch_size,this->_W_rows) = X[0][0];
        this->_activated[0][0] = this->_input[0][0] * this->W[0][0];
    }
}


void FullConnect_Layer::calc_delta(const std::shared_ptr<Layer> &next_layer) {
    if ( this->channel_num > 1 ) {
        #pragma omp parallel for
        for ( int i = 0; i < this->channel_num; ++i ) {
            this->delta[0][i]
                = next_layer->delta[0][i] *
                this->W[0][i].block(1,0,this->get_W_rows(),this->get_W_cols()).transpose();
        }
    } else {
        this->delta[0][0]
            = next_layer->delta[0][0] *
            this->W[0][0].block(1,0,this->get_W_rows(),this->get_W_cols()).transpose();
    }
}


void FullConnect_Layer::calc_differential(const std::shared_ptr<Layer> &prev_layer,
                                          const std::shared_ptr<Layer> &next_layer) {
    if ( this->channel_num > 1 ) {
        #pragma omp parallel for
        for ( int i = 0; i < this->channel_num; ++i ) {
            this->dE_dW[0][i].block(1, 0, this->_W_rows, this->_W_cols)
                = prev_layer->_activated[0][i].transpose() * this->delta[0][i];

            this->dE_dW[0][i].block(0, 0, 1, this->_W_cols) = this->delta[0][i].colwise().sum();

            this->dE_dW[0][i] /= (float)this->batch_size;
        }
    } else {
        this->dE_dW[0][0].block(1, 0, this->_W_rows, this->_W_cols)
            = prev_layer->_activated[0][0].transpose() * next_layer->delta[0][0];

        this->dE_dW[0][0].block(0, 0, 1, this->_W_cols) = next_layer->delta[0][0].colwise().sum();

        this->dE_dW[0][0] /= (float)this->batch_size;
    }
}


void FullConnect_Layer::allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->unit_num = this->_W_cols;

    this->_input.resize(1); this->_input[0].resize(this->channel_num);
    this->delta.resize(1); this->delta[0].resize(this->channel_num);
    this->_activated.resize(1); this->_activated[0].resize(channel_num);
    this->dE_dW.resize(1); this->dE_dW[0].resize(channel_num);

    for ( int i = 0; i < this->channel_num; ++i ) {
        this->_input[0][i].resize(this->batch_size, this->_W_rows+1);
        if ( this->use_bias ) {
            this->_input[0][i].block(0,0,this->batch_size,1) = MatrixXf::Ones(this->batch_size, 1);
        } else {
            this->_input[0][i].block(0,0,this->batch_size,1) = MatrixXf::Zero(this->batch_size, 1);
        }

        this->delta[0][i].resize(this->batch_size, this->_W_rows);
        this->_activated[0][i].resize(this->batch_size, this->_W_cols);
        this->dE_dW[0][i].resize(this->_W_rows+1, this->_W_cols);
    }
}


bool FullConnect_Layer::get_trainable(void) { return this->trainable; }
string FullConnect_Layer::get_type(void) { return this->type; }
bool FullConnect_Layer::get_is_tensor(void) { return this->is_tensor; }
int FullConnect_Layer::get_unit_num(void) { return this->unit_num; }
int FullConnect_Layer::get_channel_num(void) { return this->channel_num; }
int FullConnect_Layer::get_batch_size(void) { return this->batch_size; }
bool FullConnect_Layer::get_use_bias(void) { return this->use_bias; }
int FullConnect_Layer::get_W_cols(void) { return this->_W_cols; }
int FullConnect_Layer::get_W_rows(void) { return this->_W_rows; }
vector<vector <MatrixXf> > FullConnect_Layer::get_activated(void) { return this->_activated; }
vector<vector <MatrixXf> > FullConnect_Layer::get_delta(void) { return this->delta; }

void FullConnect_Layer::set_batch_size(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->allocate_memory(batch_size, prev_layer);
}
void FullConnect_Layer::set_delta(const vector<vector <MatrixXf> > delta) {
    this->delta[0][0] = delta[0][0];
}


#endif // INCLUDE_full_connect_layer_h_
