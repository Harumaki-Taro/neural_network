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

// NOTE:全結合もチャンネル化した方がいい。

class FullConnect_Layer : public Layer {
public:
    virtual void forwardprop(const vector<vector <MatrixXf> > X);
    virtual void calc_delta(const vector<vector <MatrixXf> > next_delta,
                            const vector<vector <MatrixXf> > next_bW,
                            const int nextW_rows, const int next_W_cols);
    virtual void calc_differential(const vector<vector <MatrixXf> > prev_activated);

    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf)> d_f,
                             const MatrixXf W, const MatrixXf b,
                             const bool use_bias);
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf)> d_f,
                             const int (&W_shape)[2], const bool use_bias=true,
                             const float W_min=-0.01, const float W_max=0.01,
                             const float b_min=-0.01, const float b_max=0.01);
    virtual void allocate_memory(const int batch_size);
    virtual void allocate_memory(const int batch_size, const bool use_bias_in_next_layer);

    // getter
    virtual bool get_trainable(void);
    virtual int get_batch_size(void);
    virtual bool get_use_bias(void);
    virtual vector<vector <MatrixXf> > get_bW(void);
    virtual vector<vector <MatrixXf> > get_W(void);
    virtual MatrixXf get_b(void);
    int get_W_cols(void);
    int get_W_rows(void);
    vector<vector <MatrixXf> > get_preActivate(void);
    virtual vector<vector <MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void);
    virtual function<MatrixXf(MatrixXf)> get_d_activateFunction(void);
    virtual vector<vector <MatrixXf> > get_dE_dbW(void);
    MatrixXf get_dE_dW(void);
    MatrixXf get_dE_db(void);

    // setter
    virtual void set_batch_size(const int batch_size, const bool use_bias);
    virtual void set_bW(const MatrixXf W, const MatrixXf b, const bool use_bias);
    // virtual void set_W(MatrixXf);
    // virtual void set_b(MatrixXf);
    virtual void set_delta(const vector<vector <MatrixXf> > delta);
    virtual void set_activateFunction(const function<MatrixXf(MatrixXf)> f);
    virtual void set_d_activateFunction(const function<MatrixXf(MatrixXf)> d_f);

private:
    bool trainable = true;
    // Parameters tracked during learning
    vector<vector <MatrixXf> > delta;
    // Parameters specified at first
    int batch_size;
    int _W_cols;
    int _W_rows;
    bool use_bias;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    // Storage during learning
    vector<vector <MatrixXf> > _preActivate;
};


void FullConnect_Layer::forwardprop(const vector<vector <MatrixXf> > X) {
    this->_preActivate[0][0] = X[0][0] * this->bW[0][0];
    this->_activated[0][0].block(0,1,this->batch_size,this->_preActivate[0][0].cols())
        = this->f(this->_preActivate[0][0]);
}


void FullConnect_Layer::calc_delta(const vector<vector <MatrixXf> > next_delta, const vector<vector <MatrixXf> > next_bW,
                                   const int next_W_rows, const int next_W_cols) {
    this->delta[0][0] = elemntwiseProduct(next_delta[0][0] * next_bW[0][0].block(1,0,next_W_rows,next_W_cols).transpose(),
                                    d_f(this->_activated[0][0].block(0,1,this->batch_size,_W_cols)));
}


void FullConnect_Layer::calc_differential(const vector<vector <MatrixXf> > prev_activated) {
    this->_dE_dbW[0][0] = prev_activated[0][0].transpose() * this->delta[0][0] / (float)this->batch_size;
}


void FullConnect_Layer::build_layer(const function<MatrixXf(MatrixXf)> f,
                                    const function<MatrixXf(MatrixXf)> d_f,
                                    const MatrixXf W, const MatrixXf b,
                                    const bool use_bias) {
    set_bW(W, b, use_bias);
    set_activateFunction(f);
    set_d_activateFunction(d_f);
}


void FullConnect_Layer::build_layer(const function<MatrixXf(MatrixXf)> f,
                                    const function<MatrixXf(MatrixXf)> d_f,
                                    const int (&W_shape)[2], const bool use_bias,
                                    const float W_min, const float W_max,
                                    const float b_min, const float b_max) {
    // Define weight and bias at random.
    MatrixXf W = uniform_rand(W_shape, W_min, W_max);
    MatrixXf b = uniform_rand(W_shape[1], b_min, b_max);

    this->build_layer(f,
                      d_f,
                      W, b, use_bias);
}


void FullConnect_Layer::allocate_memory(const int batch_size) {
    this->batch_size = batch_size;

    this->_preActivate.resize(1); this->_preActivate[0].resize(1);
    this->_preActivate[0][0].resize(this->batch_size, this->_W_cols);

    this->delta.resize(1); this->delta[0].resize(1);
    this->delta[0][0].resize(this->batch_size, this->_W_cols);

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->_W_cols+1);
    this->_activated[0][0].block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);

    this->_dE_dbW.resize(1); this->_dE_dbW[0].resize(1);
    this->_dE_dbW[0][0].resize(this->_W_rows+1, this->_W_cols);
}


void FullConnect_Layer::allocate_memory(const int batch_size, const bool use_bias_in_next_layer) {
    this->batch_size = batch_size;

    this->_preActivate.resize(1); this->_preActivate[0].resize(1);
    this->_preActivate[0][0].resize(this->batch_size, this->_W_cols);

    this->delta.resize(1); this->delta[0].resize(1);
    this->delta[0][0].resize(this->batch_size, this->_W_cols);

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->_W_cols+1);
    if ( use_bias_in_next_layer ) {
        this->_activated[0][0].block(0,0,this->batch_size,1) = MatrixXf::Ones(this->batch_size, 1);
    } else {
        this->_activated[0][0].block(0,0,this->batch_size,1) = MatrixXf::Zero(this->batch_size, 1);
    }

    this->_dE_dbW.resize(1); this->_dE_dbW[0].resize(1);
    this->_dE_dbW[0][0].resize(this->_W_rows+1, this->_W_cols);
}


bool FullConnect_Layer::get_trainable(void) { return this->trainable; }
int FullConnect_Layer::get_batch_size(void) { return this->batch_size; }
bool FullConnect_Layer::get_use_bias(void) { return this->use_bias; }
vector<vector <MatrixXf> > FullConnect_Layer::get_bW(void) { return this->bW; }
vector<vector <MatrixXf> > FullConnect_Layer::get_W(void) {
    vector<vector <MatrixXf> > W;
    W.resize(1); W[0].resize(1);
    W[0][0] = this->bW[0][0].block(1,0,this->_W_rows,this->_W_cols);
    return W;
}
MatrixXf FullConnect_Layer::get_b(void) {
    MatrixXf b = this->bW[0][0].block(0,0,1,this->_W_cols);
    return b;
}
int FullConnect_Layer::get_W_cols(void) { return this->_W_cols; }
int FullConnect_Layer::get_W_rows(void) { return this->_W_rows; }
vector<vector <MatrixXf> > FullConnect_Layer::get_preActivate(void) { return this->_preActivate; }
vector<vector <MatrixXf> > FullConnect_Layer::get_activated(void) { return this->_activated; }
vector<vector <MatrixXf> > FullConnect_Layer::get_delta(void) { return this->delta; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_activateFunction(void) { return this->f; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_d_activateFunction(void) { return this->d_f; }
vector<vector <MatrixXf> > FullConnect_Layer::get_dE_dbW(void) { return this->_dE_dbW; }
MatrixXf FullConnect_Layer::get_dE_dW(void) {
    return this->_dE_dbW[0][0].block(1,0,this->_W_rows,this->_W_cols);
}
MatrixXf FullConnect_Layer::get_dE_db(void) {
    return this->_dE_dbW[0][0].block(0,0,1,this->_W_cols);
}

void FullConnect_Layer::set_batch_size(const int batch_size, const bool use_bias_in_next_layer) {
    this->allocate_memory(batch_size, use_bias_in_next_layer);
}
void FullConnect_Layer::set_bW(const MatrixXf W, const MatrixXf b, const bool use_bias) {
    this->_W_cols = W.cols();
    this->_W_rows = W.rows();
    this->use_bias = use_bias;

    this->bW.resize(1); this->bW[0].resize(1);
    this->bW[0][0].resize(this->_W_rows+1, this->_W_cols);
    this->bW[0][0].block(0,0,1,this->_W_cols) = b;
    this->bW[0][0].block(1,0,this->_W_rows,this->_W_cols) = W;
}
void FullConnect_Layer::set_delta(const vector<vector <MatrixXf> > delta) {
    this->delta[0][0] = delta[0][0];
}
void FullConnect_Layer::set_activateFunction(const function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void FullConnect_Layer::set_d_activateFunction(const function<MatrixXf (MatrixXf)> d_f) {
    this->d_f = d_f;
}


#endif // INCLUDE_full_connect_layer_h_
