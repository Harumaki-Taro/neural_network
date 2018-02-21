#ifndef INCLUDE_full_connect_layer_h_
#define INCLUDE_full_connect_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class FullConnect_Layer : public Layer {
public:
    virtual void forwardprop(MatrixXf);
    virtual void calc_delta(MatrixXf, MatrixXf, MatrixXf);
    virtual void calc_differential(MatrixXf);

    virtual void build_layer(MatrixXf, MatrixXf, bool,
                     function<MatrixXf(MatrixXf)>,
                     function<MatrixXf(MatrixXf)>);
    virtual void allocate_memory(int, bool);
    virtual void allocate_memory(int);
    // setter
    virtual void set_bW(MatrixXf, MatrixXf, bool);
    virtual void set_activateFunction(function<MatrixXf(MatrixXf)>);
    virtual void set_d_activateFunction(function<MatrixXf(MatrixXf)>);
    // getter
    virtual MatrixXf get_bW(void);
    virtual bool get_use_bias(void);
    virtual function<MatrixXf(MatrixXf)> get_f(void);
    virtual function<MatrixXf(MatrixXf)> get_d_f(void);
    virtual MatrixXf get_preActivate(void);
    virtual MatrixXf get_activated_(void);
    virtual MatrixXf get_delta(void);
    virtual MatrixXf get_dE_dbW(void);
    virtual int get_batch_size(void);
    virtual MatrixXf get_W(void);
    virtual MatrixXf get_b(void);
    virtual MatrixXf get_dE_dW(void);
    virtual MatrixXf get_dE_db(void);

    // 一時的な対処
    // MatrixXf activated_;
    // MatrixXf W;
    // MatrixXf delta;
    // function<MatrixXf(MatrixXf)> d_f;
    // MatrixXf bW;
    // MatrixXf dE_dbW;
    // bool use_bias;

private:
    // ユーザーにより初期値指定
    function<MatrixXf(MatrixXf)> f;
    // 計算結果の格納
    MatrixXf preActivate;

    int batch_size;

    MatrixXf b;
};


void FullConnect_Layer::forwardprop(MatrixXf X) {
    preActivate = X * bW;
    activated_.block(0,1,batch_size,preActivate.cols()) = f(preActivate);
}


void FullConnect_Layer::calc_delta(MatrixXf next_delta, MatrixXf next_bW, MatrixXf next_W) {
    delta = elemntwiseProduct(next_delta * next_bW.block(1,0,next_W.rows(),next_W.cols()).transpose(),
                                     d_f(activated_.block(0,1,batch_size,W.cols())));
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
    preActivate.resize(batch_size, W.cols());
    delta.resize(batch_size, W.cols());

    activated_.resize(batch_size, W.cols()+1);
    if ( use_bias_in_next_layer ) {
        activated_.block(0,0,batch_size,1) = MatrixXf::Ones(batch_size, 1);
    } else {
        activated_.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    }
    dE_dbW.resize(W.rows()+1, W.cols());
}


void FullConnect_Layer::allocate_memory(int batch_size) {
    this->batch_size = batch_size;
    preActivate.resize(batch_size, W.cols());
    delta.resize(batch_size, W.cols());

    activated_.resize(batch_size, W.cols()+1);
    activated_.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    dE_dbW.resize(W.rows()+1, W.cols());
}


void FullConnect_Layer::set_bW(MatrixXf b, MatrixXf W, bool use_bias) {
    this->W = W;
    this->b = b;
    this->use_bias = use_bias;
    this->bW.resize(W.rows()+1, W.cols());
    this->bW.block(0,0,1,W.cols()) = b;
    this->bW.block(1,0,W.rows(),W.cols()) = W;
}


void FullConnect_Layer::set_activateFunction(function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void FullConnect_Layer::set_d_activateFunction(function<MatrixXf (MatrixXf)> d_f) {
    this->d_f = d_f;
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
    this->W = this->bW.block(1,0,W.rows(),W.cols());
    return this->W;
}
MatrixXf FullConnect_Layer::get_b(void) {
    this->b = this->bW.block(0,0,1,W.cols());
    return this->b;
}
MatrixXf FullConnect_Layer::get_dE_dW(void) {
    return this->dE_dbW.block(1,0,W.rows(),W.cols());
}
MatrixXf FullConnect_Layer::get_dE_db(void) {
    return this->dE_dbW.block(0,0,1,W.cols());
}


#endif // INCLUDE_full_connect_layer_h_
