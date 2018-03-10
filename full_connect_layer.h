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
    virtual void forwardprop(MatrixXf);
    virtual void calc_delta(MatrixXf, MatrixXf, int, int);
    virtual void calc_differential(MatrixXf);

    virtual void build_layer(MatrixXf, MatrixXf, bool,
                     function<MatrixXf(MatrixXf)>,
                     function<MatrixXf(MatrixXf)>);
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf)> d_f,
                             const int (&W_shape)[2], const bool use_bias=true,
                             const float W_max=1.f, const float W_min=-1.f,
                             const float b_max=1.f, const float b_min=-1.f);
    virtual void allocate_memory(int);
    virtual void allocate_memory(int, bool);

    // getter
    virtual bool get_trainable(void);
    virtual int get_batch_size(void);
    virtual bool get_use_bias(void);
    virtual MatrixXf get_bW(void);
    virtual MatrixXf get_W(void);
    virtual MatrixXf get_b(void);
    int get_W_cols(void);
    int get_W_rows(void);
    MatrixXf get_preActivate(void);
    virtual MatrixXf get_activated(void);
    virtual MatrixXf get_delta(void);
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void);
    virtual function<MatrixXf(MatrixXf)> get_d_activateFunction(void);
    virtual MatrixXf get_dE_dbW(void);
    MatrixXf get_dE_dW(void);
    MatrixXf get_dE_db(void);

    // setter
    virtual void set_batch_size(int, bool);
    virtual void set_bW(MatrixXf, MatrixXf, bool);
    // virtual void set_W(MatrixXf);
    // virtual void set_b(MatrixXf);
    virtual void set_delta(MatrixXf);
    virtual void set_activateFunction(function<MatrixXf(MatrixXf)>);
    virtual void set_d_activateFunction(function<MatrixXf(MatrixXf)>);

private:
    bool trainable = true;
    // Parameters tracked during learning
    MatrixXf delta;
    // Parameters specified at first
    int batch_size;
    int _W_cols;
    int _W_rows;
    bool use_bias;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    // Storage during learning
    MatrixXf _preActivate;
};


void FullConnect_Layer::forwardprop(MatrixXf X) {
    this->_preActivate = X * bW;
    this->_activated.block(0,1,this->batch_size,this->_preActivate.cols()) = f(this->_preActivate);
}


void FullConnect_Layer::calc_delta(MatrixXf next_delta, MatrixXf next_bW,
                                   int next_W_rows, int next_W_cols) {
    this->delta = elemntwiseProduct(next_delta * next_bW.block(1,0,next_W_rows,next_W_cols).transpose(),
                                    d_f(this->_activated.block(0,1,this->batch_size,_W_cols)));
}


void FullConnect_Layer::calc_differential(MatrixXf prev_activated_) {
    this->_dE_dbW = prev_activated_.transpose() * this->delta;
}


void FullConnect_Layer::build_layer(MatrixXf b, MatrixXf W, bool use_bias,
                                    function<MatrixXf(MatrixXf)> f,
                                    function<MatrixXf(MatrixXf)> d_f) {
    set_bW(b, W, use_bias);
    set_activateFunction(f);
    set_d_activateFunction(d_f);
}


void FullConnect_Layer::build_layer(const function<MatrixXf(MatrixXf)> f,
                                    const function<MatrixXf(MatrixXf)> d_f,
                                    const int (&W_shape)[2], const bool use_bias,
                                    const float W_max, const float W_min,
                                    const float b_max, const float b_min) {
    // Define weight and bias at random.
    MatrixXf W = uniform_rand(W_shape, W_max, W_min);
    MatrixXf b = uniform_rand(W_shape[1], b_max, b_min);

    this->build_layer(b, W, use_bias,
                      f,
                      d_f);
}


void FullConnect_Layer::allocate_memory(int batch_size) {
    this->batch_size = batch_size;
    this->_preActivate.resize(this->batch_size, this->_W_cols);
    this->delta.resize(this->batch_size, this->_W_cols);

    this->_activated.resize(this->batch_size, this->_W_cols+1);
    this->_activated.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    this->_dE_dbW.resize(this->_W_rows+1, this->_W_cols);
}


void FullConnect_Layer::allocate_memory(int batch_size, bool use_bias_in_next_layer) {
    this->batch_size = batch_size;
    this->_preActivate.resize(this->batch_size, this->_W_cols);
    this->delta.resize(this->batch_size, this->_W_cols);

    this->_activated.resize(this->batch_size, this->_W_cols+1);
    if ( use_bias_in_next_layer ) {
        this->_activated.block(0,0,this->batch_size,1) = MatrixXf::Ones(this->batch_size, 1);
    } else {
        this->_activated.block(0,0,this->batch_size,1) = MatrixXf::Zero(this->batch_size, 1);
    }

    this->_dE_dbW.resize(this->_W_rows+1, this->_W_cols);
}


bool FullConnect_Layer::get_trainable(void) { return this->trainable; }
int FullConnect_Layer::get_batch_size(void) { return this->batch_size; }
bool FullConnect_Layer::get_use_bias(void) { return this->use_bias; }
MatrixXf FullConnect_Layer::get_bW(void) { return this->bW; }
MatrixXf FullConnect_Layer::get_W(void) {
    MatrixXf W = this->bW.block(1,0,this->_W_rows,this->_W_cols);
    return W;
}
MatrixXf FullConnect_Layer::get_b(void) {
    MatrixXf b = this->bW.block(0,0,1,this->_W_cols);
    return b;
}
int FullConnect_Layer::get_W_cols(void) { return this->_W_cols; }
int FullConnect_Layer::get_W_rows(void) { return this->_W_rows; }
MatrixXf FullConnect_Layer::get_preActivate(void) { return this->_preActivate; }
MatrixXf FullConnect_Layer::get_activated(void) { return this->_activated; }
MatrixXf FullConnect_Layer::get_delta(void) { return this->delta; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_activateFunction(void) { return this->f; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_d_activateFunction(void) { return this->d_f; }
MatrixXf FullConnect_Layer::get_dE_dbW(void) { return this->_dE_dbW; }
MatrixXf FullConnect_Layer::get_dE_dW(void) {
    return this->_dE_dbW.block(1,0,this->_W_rows,this->_W_cols);
}
MatrixXf FullConnect_Layer::get_dE_db(void) {
    return this->_dE_dbW.block(0,0,1,this->_W_cols);
}

void FullConnect_Layer::set_batch_size(int batch_size, bool use_bias_in_next_layer) {
    this->allocate_memory(batch_size, use_bias_in_next_layer);
}
void FullConnect_Layer::set_bW(MatrixXf b, MatrixXf W, bool use_bias) {
    this->_W_cols = W.cols();
    this->_W_rows = W.rows();
    this->use_bias = use_bias;
    this->bW.resize(this->_W_rows+1, this->_W_cols);
    this->bW.block(0,0,1,this->_W_cols) = b;
    this->bW.block(1,0,this->_W_rows,this->_W_cols) = W;
}
void FullConnect_Layer::set_delta(MatrixXf delta) {
    this->delta = delta;
}
void FullConnect_Layer::set_activateFunction(function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void FullConnect_Layer::set_d_activateFunction(function<MatrixXf (MatrixXf)> d_f) {
    this->d_f = d_f;
}


#endif // INCLUDE_full_connect_layer_h_
