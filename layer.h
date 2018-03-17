#ifndef INCLUDE_layer_h_
#define INCLUDE_layer_h_

#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "my_math.h"

using std::function;
using std::cout;
using std::endl;
using std::shared_ptr;
using Eigen::MatrixXf;


class Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X) { cout << "使用禁止a2" << endl; exit(1); }
    virtual void forwardprop(const MatrixXf X) { cout << "使用禁止a3" << endl; exit(1); }

    virtual void calc_delta(const MatrixXf pred, const MatrixXf label) { cout << "使用禁止c" << endl; exit(1); }
    virtual void calc_delta(const shared_ptr<Layer> &next_layer) { cout << "使用禁止c5" << endl; exit(1); }
    virtual void calc_delta(const shared_ptr<Layer> &next_layer,
                            const shared_ptr<Layer> &prev_layer) { cout << "使用禁止c6" << endl; exit(1); }

    virtual void calc_differential(const shared_ptr<Layer> &prev_layer,
                                   const shared_ptr<Layer> &next_layer) { cout << "使用禁止d4" << endl; exit(1); }
    virtual void calc_differential(const shared_ptr<Layer> &prev_layer) { cout << "使用禁止d5" << endl; exit(1); }

    virtual void allocate_memory(const int batch_size) { cout << "使用禁止g" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size,
                                 const int example_size) { cout << "使用禁止i" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size, const int prev_cols, const int prev_rows) { cout << "使用禁止i2" << endl; exit(1); }

    // getter
    virtual bool get_trainable(void) { cout << "使用禁止j" << endl; exit(1); return 1; }
    virtual string get_type(void) { cout << "使用禁止j2" << endl; exit(1); return "0"; }
    virtual int get_batch_size(void) { cout << "使用禁止k" << endl; exit(1); return 1; }
    virtual bool get_use_bias(void) { cout << "使用禁止l" << endl; exit(1); return false; }
    virtual int get_W_cols(void) { cout << "使用禁止p" << endl; exit(1); return 1; }
    virtual int get_W_rows(void) { cout << "使用禁止q" << endl; exit(1); return 1; }
    virtual vector< vector<MatrixXf> > get_activated(void) {
        cout << "使用禁止r2" << endl; exit(1);
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    virtual vector<vector <MatrixXf> > get_delta(void) {
        cout << "使用禁止r2" << endl; exit(1);
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void) { cout << "使用禁止t" << endl; exit(1); return identity; }
    virtual function<MatrixXf(MatrixXf)> get_d_activateFunction(void) { cout << "使用禁止u" << endl; exit(1); return identity; }
    virtual vector<int> get_input_map_shape(void) {cout << "使用禁止v2" << endl; exit(1); vector<int> tmp; tmp.resize(1); tmp[0] = 0; return tmp; }
    virtual vector<int> get_output_map_shape(void) {cout << "使用禁止v3" << endl; exit(1); vector<int> tmp; tmp.resize(1); tmp[0] = 0; return tmp; }

    // setter
    virtual void set_batch_size(const int batch_size) { cout << "使用禁止w" << endl; exit(1); }
    virtual void set_batch_size(const int batch_size, const int prev_cols, const int prev_rows) { cout << "使用禁止x2" << endl; exit(1); }
    virtual void set_bW(const MatrixXf W, const MatrixXf b, const bool use_bias) { cout << "使用禁止y" << endl; exit(1); }
    virtual void set_delta(const MatrixXf delta) { cout << "使用禁止z" << endl; exit(1); }
    virtual void set_delta(const vector< vector<MatrixXf> > delta) { cout << "使用禁止z2" << endl; exit(1); }
    virtual void set_activateFunction(const function<MatrixXf(MatrixXf)> f) { cout << "使用禁止aa" << endl; exit(1); }
    virtual void set_d_activateFunction(const function<MatrixXf(MatrixXf)> d_f) { cout << "使用禁止ab" << endl; exit(1); }
    virtual int get_channel_num(void) { cout << "使用禁止ac" << endl; exit(1); return 1; }
    virtual int get_prev_channel_num(void) { cout << "使用禁止ad" << endl; exit(1); return 1; }
    virtual vector<int>  get_filter_shape(void) { cout << "使用禁止ae" << endl; exit(1); vector<int> tmp; tmp.resize(1); tmp[0] = 0; return tmp; }
    // 子クラスのprivateへ移行予定
    vector<vector <MatrixXf> > W;   // NOTE:convolutionしか使ってない問題
    vector< vector<MatrixXf> > dE_dW;   // NOTE:convolutionしか使ってない問題
    MatrixXf dE_db; // NOTE:convolutionしか使ってない問題
    MatrixXf b; // NOTE:convolutionしか使ってない問題
    vector<vector <MatrixXf> > _activated;
    vector<vector <MatrixXf> > delta;
};

#endif // INCLUDE_layer_h_
