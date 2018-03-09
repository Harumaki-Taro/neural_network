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
    virtual void forwardprop(MatrixXf) { cout << "使用禁止a" << endl; exit(1); }
    virtual void calc_delta(MatrixXf, MatrixXf, int, int) { cout << "使用禁止b" << endl; exit(1); }
    virtual void calc_delta(MatrixXf, MatrixXf) { cout << "使用禁止c" << endl; exit(1); }
    virtual void calc_differential(MatrixXf) { cout << "使用禁止d" << endl; exit(1); }
    virtual void build_layer(MatrixXf, MatrixXf, bool,
                             function<MatrixXf(MatrixXf)>,
                             function<MatrixXf(MatrixXf)>) { cout << "使用禁止e" << endl; exit(1); }
    virtual void build_layer(int,
                             function<MatrixXf(MatrixXf)>,
                             function<MatrixXf(MatrixXf, MatrixXf)>) { cout << "使用禁止f" << endl; exit(1); }
    virtual void allocate_memory(int, int, bool) { cout << "使用禁止g" << endl; exit(1); }
    virtual void allocate_memory(int, bool) { cout << "使用禁止h" << endl; exit(1); }
    virtual void allocate_memory(int) { cout << "使用禁止i" << endl; exit(1); }
    // setter
    virtual void set_bW(MatrixXf, MatrixXf, bool) { cout << "使用禁止j" << endl; exit(1); }
    virtual void set_delta(MatrixXf) { cout << "使用禁止k" << endl; exit(1); }
    // getter
    virtual MatrixXf get_bW(void) { cout << "使用禁止l" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual bool get_use_bias(void) { cout << "使用禁止m" << endl; exit(1); return false; }
    virtual MatrixXf get_activated_(void) { cout << "使用禁止n" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_delta(void) { cout << "使用禁止o" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_dE_dbW(void) { cout << "使用禁止p" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual int get_batch_size(void) { cout << "使用禁止q" << endl; exit(1); return 1; }
    virtual MatrixXf get_W(void) { cout << "使用禁止r" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_b(void) { cout << "使用禁止s" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual int get_W_cols(void) { cout << "使用禁止t" << endl; exit(1); return 1; }
    virtual int get_W_rows(void) { cout << "使用禁止u" << endl; exit(1); return 1; }

    // 子クラスのprivateへ移行予定
    MatrixXf bW;
    MatrixXf activated_;
};

#endif // INCLUDE_layer_h_
