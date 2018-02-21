#ifndef INCLUDE_layer_h_
#define INCLUDE_layer_h_

#include <iostream>
#include <functional>
#include "Eigen/Core"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class Layer {
public:
    virtual void forwardprop(MatrixXf) = 0;
    virtual void calc_delta(MatrixXf, MatrixXf, MatrixXf) = 0;
    virtual void calc_differential(MatrixXf) = 0;

    virtual void build_layer(MatrixXf, MatrixXf, bool,
                     function<MatrixXf(MatrixXf)>,
                     function<MatrixXf(MatrixXf)>) = 0;
    virtual void allocate_memory(int, bool) = 0;
    virtual void allocate_memory(int) = 0;
    // setter
    virtual void set_bW(MatrixXf, MatrixXf, bool) = 0;
    virtual void set_activateFunction(function<MatrixXf(MatrixXf)>) = 0;
    virtual void set_d_activateFunction(function<MatrixXf(MatrixXf)>) = 0;
    // getter
    virtual MatrixXf get_bW(void) = 0;
    virtual bool get_use_bias(void) = 0;
    virtual function<MatrixXf(MatrixXf)> get_f(void) = 0;
    virtual function<MatrixXf(MatrixXf)> get_d_f(void) = 0;
    virtual MatrixXf get_preActivate(void) = 0;
    virtual MatrixXf get_activated_(void) = 0;
    virtual MatrixXf get_delta(void) = 0;
    virtual MatrixXf get_dE_dbW(void) = 0;
    virtual int get_batch_size(void) = 0;
    virtual MatrixXf get_W(void) = 0;
    virtual MatrixXf get_b(void) = 0;
    virtual MatrixXf get_dE_dW(void) = 0;
    virtual MatrixXf get_dE_db(void) = 0;

    // 一時的な対処
    MatrixXf activated_;
    MatrixXf W;
    MatrixXf delta;
    function<MatrixXf(MatrixXf)> d_f;
    MatrixXf bW;
    MatrixXf dE_dbW;
    bool use_bias;
};

#endif // INCLUDE_layer_h_
