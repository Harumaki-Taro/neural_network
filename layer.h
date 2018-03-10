#ifndef INCLUDE_layer_h_
#define INCLUDE_layer_h_

#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "my_math.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class Layer {
public:
    virtual void forwardprop(const MatrixXf X) { cout << "使用禁止a" << endl; exit(1); }
    virtual void calc_delta(const MatrixXf next_delta, const MatrixXf next_bW,
                            const int nextW_rows, const int nextW_cols) { cout << "使用禁止b" << endl; exit(1); }
    virtual void calc_delta(MatrixXf, MatrixXf) { cout << "使用禁止c" << endl; exit(1); }
    virtual void calc_differential(MatrixXf) { cout << "使用禁止d" << endl; exit(1); }
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf)> d_f,
                             const MatrixXf W, const MatrixXf b,
                             const bool use_bias) { cout << "使用禁止e" << endl; exit(1); }
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf)> d_f,
                             const int (&W_shape)[2], const bool use_bias=true,
                             const float W_min=-1.f, const float W_max=1.f,
                             const float b_min=-1.f, const float b_max=1.f) { cout << "使用禁止e2" << endl; exit(1); }
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                             const int class_num) { cout << "使用禁止f" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size) { cout << "使用禁止g" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size,
                                 const bool use_bias_in_next_layer) { cout << "使用禁止h" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size,
                                 const int example_size,
                                 const bool use_bias) { cout << "使用禁止i" << endl; exit(1); }
    // getter
    virtual bool get_trainable(void) { cout << "使用禁止j" << endl; exit(1); return 1; }
    virtual int get_batch_size(void) { cout << "使用禁止k" << endl; exit(1); return 1; }
    virtual bool get_use_bias(void) { cout << "使用禁止l" << endl; exit(1); return false; }
    virtual MatrixXf get_bW(void) { cout << "使用禁止m" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_W(void) { cout << "使用禁止n" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_b(void) { cout << "使用禁止o" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual int get_W_cols(void) { cout << "使用禁止p" << endl; exit(1); return 1; }
    virtual int get_W_rows(void) { cout << "使用禁止q" << endl; exit(1); return 1; }
    virtual MatrixXf get_activated(void) { cout << "使用禁止r" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_delta(void) { cout << "使用禁止s" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void) { cout << "使用禁止t" << endl; exit(1); return identity; }
    virtual function<MatrixXf(MatrixXf)> get_d_activateFunction(void) { cout << "使用禁止u" << endl; exit(1); return identity; }
    virtual MatrixXf get_dE_dbW(void) { cout << "使用禁止v" << endl; exit(1); return MatrixXf::Zero(1,1); }
    // setter
    virtual void set_batch_size(const int batch_size) { cout << "使用禁止w" << endl; exit(1); }
    virtual void set_batch_size(const int batch_size,
                                const bool use_bias_in_next_layer) { cout << "使用禁止x" << endl; exit(1); }
    virtual void set_bW(const MatrixXf W, const MatrixXf b, const bool use_bias) { cout << "使用禁止y" << endl; exit(1); }
    virtual void set_delta(const MatrixXf delta) { cout << "使用禁止z" << endl; exit(1); }
    virtual void set_activateFunction(const function<MatrixXf(MatrixXf)> f) { cout << "使用禁止aa" << endl; exit(1); }
    virtual void set_d_activateFunction(const function<MatrixXf(MatrixXf)> d_f) { cout << "使用禁止ab" << endl; exit(1); }
    // 子クラスのprivateへ移行予定
    MatrixXf _dE_dbW;
    MatrixXf bW;
    MatrixXf _activated;
};

#endif // INCLUDE_layer_h_
