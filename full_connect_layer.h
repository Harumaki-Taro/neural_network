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
    virtual void allocate_memory(int, bool);
    virtual void allocate_memory(int);
    // setter
    virtual void set_bW(MatrixXf, MatrixXf, bool);
    void set_activateFunction(function<MatrixXf(MatrixXf)>);
    void set_d_activateFunction(function<MatrixXf(MatrixXf)>);
    virtual void set_delta(MatrixXf);
    // getter
    virtual MatrixXf get_bW(void);
    virtual bool get_use_bias(void);
    function<MatrixXf(MatrixXf)> get_f(void);
    function<MatrixXf(MatrixXf)> get_d_f(void);
    MatrixXf get_preActivate(void);
    virtual MatrixXf get_activated_(void);
    virtual MatrixXf get_delta(void);
    virtual MatrixXf get_dE_dbW(void);
    virtual int get_batch_size(void);
    virtual MatrixXf get_W(void);
    virtual MatrixXf get_b(void);
    MatrixXf get_dE_dW(void);
    MatrixXf get_dE_db(void);
    int get_W_cols(void);
    int get_W_rows(void);
    virtual bool get_trainable(void);

private:
    bool trainable = true;
    // Parameters tracked during learning
    MatrixXf delta;
    // Parameters specified at first
    int batch_size;
    int W_cols;
    int W_rows;
    bool use_bias;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    // Storage during learning
    MatrixXf preActivate;
    // 今後削除予定。
    MatrixXf W;
    MatrixXf b;
};


void FullConnect_Layer::forwardprop(MatrixXf X) {
    preActivate = X * bW;
    this->activated_.block(0,1,batch_size,preActivate.cols()) = f(preActivate);
}


void FullConnect_Layer::calc_delta(MatrixXf next_delta, MatrixXf next_bW,
                                   int next_W_rows, int next_W_cols) {
    delta = elemntwiseProduct(next_delta * next_bW.block(1,0,next_W_rows,next_W_cols).transpose(),
                              d_f(activated_.block(0,1,batch_size,W_cols)));
}


void FullConnect_Layer::calc_differential(MatrixXf prev_activated_) {
    dE_dbW = prev_activated_.transpose() * delta;
}


void FullConnect_Layer::build_layer(MatrixXf b, MatrixXf W, bool use_bias,
                                    function<MatrixXf(MatrixXf)> f,
                                    function<MatrixXf(MatrixXf)> d_f) {
    set_bW(b, W, use_bias);
    set_activateFunction(f);
    set_d_activateFunction(d_f);
    }


void FullConnect_Layer::allocate_memory(int batch_size, bool use_bias_in_next_layer) {
    this->batch_size = batch_size;
    preActivate.resize(batch_size, W_cols);
    delta.resize(batch_size, W_cols);

    activated_.resize(batch_size, W_cols+1);
    if ( use_bias_in_next_layer ) {
        activated_.block(0,0,batch_size,1) = MatrixXf::Ones(batch_size, 1);
    } else {
        activated_.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    }
    dE_dbW.resize(W_rows+1, W_cols);
    shared_ptr<MatrixXf> dE_dbW_ptr(new MatrixXf(this->dE_dbW));
}


void FullConnect_Layer::allocate_memory(int batch_size) {
    this->batch_size = batch_size;
    preActivate.resize(batch_size, W_cols);
    delta.resize(batch_size, W_cols);

    activated_.resize(batch_size, W_cols+1);
    activated_.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    dE_dbW.resize(W_rows+1, W_cols);
    shared_ptr<MatrixXf> dE_dbW_ptr(new MatrixXf(this->dE_dbW));
}


void FullConnect_Layer::set_bW(MatrixXf b, MatrixXf W, bool use_bias) {
    this->W = W;
    this->b = b;
    this->W_cols = W.cols();
    this->W_rows = W.rows();
    this->use_bias = use_bias;
    this->bW.resize(W_rows+1, W_cols);
    this->bW.block(0,0,1,W_cols) = b;
    this->bW.block(1,0,W_rows,W_cols) = W;
}


void FullConnect_Layer::set_activateFunction(function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void FullConnect_Layer::set_d_activateFunction(function<MatrixXf (MatrixXf)> d_f) {
    this->d_f = d_f;
}
void FullConnect_Layer::set_delta(MatrixXf delta) {
    this->delta = delta;
}

MatrixXf FullConnect_Layer::get_bW(void) { return this->bW; }
bool FullConnect_Layer::get_use_bias(void) { return this->use_bias; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_f(void) { return this->f; }
function<MatrixXf(MatrixXf)> FullConnect_Layer::get_d_f(void) { return this->d_f; }
MatrixXf FullConnect_Layer::get_preActivate(void) { return this->preActivate; }
MatrixXf FullConnect_Layer::get_activated_(void) { return this->activated_; }
MatrixXf FullConnect_Layer::get_delta(void) { return this->delta; }
MatrixXf FullConnect_Layer::get_dE_dbW(void) { return this->dE_dbW; }
int FullConnect_Layer::get_batch_size(void) { return this->batch_size; }
MatrixXf FullConnect_Layer::get_W(void) {
    this->W = this->bW.block(1,0,W_rows,W_cols);
    return this->W;
}
MatrixXf FullConnect_Layer::get_b(void) {
    this->b = this->bW.block(0,0,1,W_cols);
    return this->b;
}
MatrixXf FullConnect_Layer::get_dE_dW(void) {
    return this->dE_dbW.block(1,0,W_rows,W_cols);
}
MatrixXf FullConnect_Layer::get_dE_db(void) {
    return this->dE_dbW.block(0,0,1,W_cols);
}
int FullConnect_Layer::get_W_cols(void) { return this->W_cols; }
int FullConnect_Layer::get_W_rows(void) { return this->W_rows; }
bool FullConnect_Layer::get_trainable(void) { return this->trainable; }
// shared_ptr<MatrixXf> FullConnect_Layer::get_dE_dbW_ptr(void) { return this->dE_dbW_ptr; }


#endif // INCLUDE_full_connect_layer_h_
