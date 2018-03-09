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
    virtual void forwardprop(MatrixXf) { cout << "使用禁止" << endl; exit(1); }
    virtual void calc_delta(MatrixXf, MatrixXf, int, int) { cout << "使用禁止" << endl; exit(1); }
    virtual void calc_delta(MatrixXf, MatrixXf) { cout << "使用禁止" << endl; exit(1); }
    virtual void calc_differential(MatrixXf) { cout << "使用禁止" << endl; exit(1); }
    virtual void build_layer(MatrixXf, MatrixXf, bool,
                             function<MatrixXf(MatrixXf)>,
                             function<MatrixXf(MatrixXf)>) { cout << "使用禁止" << endl; exit(1); }
    virtual void build_layer(int,
                             function<MatrixXf(MatrixXf)>,
                             function<MatrixXf(MatrixXf, MatrixXf)>) { cout << "使用禁止" << endl; exit(1); }
    virtual void allocate_memory(int, int, bool) { cout << "使用禁止" << endl; exit(1); }
    virtual void allocate_memory(int, bool) { cout << "使用禁止" << endl; exit(1); }
    virtual void allocate_memory(int) { cout << "使用禁止" << endl; exit(1); }
    // setter
    virtual void set_bW(MatrixXf, MatrixXf, bool) { cout << "使用禁止" << endl; exit(1); }
    virtual void set_delta(MatrixXf) { cout << "使用禁止" << endl; exit(1); }
    // getter
    virtual MatrixXf get_bW(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual bool get_use_bias(void) { cout << "使用禁止" << endl; exit(1); return false; }
    virtual MatrixXf get_activated_(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_delta(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_dE_dbW(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual int get_batch_size(void) { cout << "使用禁止" << endl; exit(1); return 1; }
    virtual MatrixXf get_W(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual MatrixXf get_b(void) { cout << "使用禁止" << endl; exit(1); return MatrixXf::Zero(1,1); }

    MatrixXf W;
    MatrixXf bW;
    int W_cols;
    int W_rows;
    MatrixXf activated_;
    MatrixXf delta;
    function<MatrixXf(MatrixXf)> d_f;
    MatrixXf dE_dbW;
    bool use_bias;
};

#endif // INCLUDE_layer_h_
