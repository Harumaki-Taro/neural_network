#ifndef INCLUDE_full_connect_layer_h_
#define INCLUDE_full_connect_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include "Eigen/Core"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class FullConnect_Layer {
public:
    void forwardprop(MatrixXf);
    void calc_delta(FullConnect_Layer);
    void calc_differential(FullConnect_Layer);

    void build_layer(MatrixXf, MatrixXf, bool,
                     function<MatrixXf(MatrixXf)>,
                     function<MatrixXf(MatrixXf)>);
    void allocate_memory(int, FullConnect_Layer);
    void allocate_memory(int);
    // setter
    void set_bW(MatrixXf, MatrixXf, bool);
    void set_activateFunction(function<MatrixXf(MatrixXf)>);
    void set_d_activateFunction(function<MatrixXf(MatrixXf)>);
    // getter
    MatrixXf get_bW(void);
    bool get_use_bias(void);
    function<MatrixXf(MatrixXf)> get_f(void);
    function<MatrixXf(MatrixXf)> get_d_f(void);
    MatrixXf get_preActivate(void);
    MatrixXf get_activated_(void);
    MatrixXf get_delta(void);
    MatrixXf get_dE_dbW(void);
    int get_batch_size(void);
    MatrixXf get_W(void);
    MatrixXf get_b(void);
    MatrixXf get_dE_dW(void);
    MatrixXf get_dE_db(void);

    // 一時的な対処
    MatrixXf activated_;
    MatrixXf W;
    MatrixXf delta;
    function<MatrixXf(MatrixXf)> d_f;
    MatrixXf bW;
    MatrixXf dE_dbW;

private:
    // ユーザーにより初期値指定
    bool use_bias;
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


void FullConnect_Layer::calc_delta(FullConnect_Layer next_layer) {
    delta = elemntwiseProduct(next_layer.delta * next_layer.bW.block(1,0,next_layer.W.rows(),next_layer.W.cols()).transpose(),
                                     d_f(activated_.block(0,1,batch_size,W.cols())));
}


void FullConnect_Layer::calc_differential(FullConnect_Layer prev_layer) {
    dE_dbW = prev_layer.activated_.transpose() * delta;
}


void FullConnect_Layer::build_layer(MatrixXf b, MatrixXf W, bool use_bias,
                                    function<MatrixXf(MatrixXf)> f,
                                    function<MatrixXf(MatrixXf)> d_f) {
    set_bW(b, W, use_bias);
    set_activateFunction(f);
    set_d_activateFunction(d_f);
    }


void FullConnect_Layer::allocate_memory(int batch_size, FullConnect_Layer next_layer) {
    this->batch_size = batch_size;
    preActivate.resize(batch_size, W.cols());
    delta.resize(batch_size, W.cols());

    activated_.resize(batch_size, W.cols()+1);
    if ( next_layer.use_bias ) {
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
