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
    // virtual void forwardprop(const MatrixXf X) { cout << "使用禁止a" << endl; exit(1); }
    virtual void forwardprop(const vector< vector<MatrixXf> > X) { cout << "使用禁止a2" << endl; exit(1); }
    virtual void forwardprop(const MatrixXf X) { cout << "使用禁止a3" << endl; exit(1); }
    virtual void calc_delta(const vector<vector <MatrixXf> > next_delta,
                            const vector<vector <MatrixXf> > next_bW,
                            const int nextW_rows, const int nextW_cols) { cout << "使用禁止b" << endl; exit(1); }
    virtual void calc_delta(const MatrixXf, const MatrixXf) { cout << "使用禁止c" << endl; exit(1); }
    virtual void calc_delta(const vector< vector<MatrixXf> > next_delta) { cout << "使用禁止c2" << endl; exit(1); }
    virtual void calc_delta(const MatrixXf next_delta) { cout << "使用禁止c3" << endl; exit(1); }
    virtual void calc_delta(const vector<vector<MatrixXf> > next_delta,
                            const vector<vector<MatrixXf> > prev_activated) { cout << "使用禁止c4" << endl; exit(1); }
    virtual void calc_differential(MatrixXf) { cout << "使用禁止d" << endl; exit(1); }
    virtual void calc_differential(const vector< vector<MatrixXf> > prev_activated) { cout << "使用禁止d2" << endl; exit(1); }
    virtual void calc_differential(const vector< vector<MatrixXf> > prev_activated,
                                   const vector< vector<MatrixXf> > next_delta) { cout << "使用禁止d3" << endl; exit(1); }
    // virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
    //                          const function<MatrixXf(MatrixXf)> d_f,
    //                          const MatrixXf W, const MatrixXf b,
    //                          const bool use_bias) { cout << "使用禁止e" << endl; exit(1); }
    // virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
    //                          const function<MatrixXf(MatrixXf)> d_f,
    //                          const int (&W_shape)[2], const bool use_bias=true,
    //                          const float W_min=-1.f, const float W_max=1.f,
    //                          const float b_min=-1.f, const float b_max=1.f) { cout << "使用禁止e2" << endl; exit(1); }
    // virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
    //                          const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
    //                          const int class_num) { cout << "使用禁止f" << endl; exit(1); }
    // virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
    //                          const function<MatrixXf(MatrixXf)> d_f,
    //                          const int prev_ch, const int ch,
    //                          const int filter_height, const int filter_width,
    //                          const int stlide_height, const int stlide_width,
    //                          const int padding_height, const int padding_width,
    //                          const float W_min, const float W_max,
    //                          const float b_min, const float b_max) { cout << "使用禁止f2" << endl; exit(1); }
    // virtual void build_layer(const int channel_num,
    //                          const int filter_height, const int filter_width,
    //                          const int stlide_height=1, const int stlide_width=1,
    //                          const int padding_height=0, const int padding_width=0) { cout << "使用禁止f4" << endl; exit(1); }
    // virtual void build_layer(const int channel_num, const int height, const int width) { cout << "使用禁止f3" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size) { cout << "使用禁止g" << endl; exit(1); }
    // virtual void allocate_memory(const int batch_size) { cout << "使用禁止h" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size,
                                 const int example_size) { cout << "使用禁止i" << endl; exit(1); }
    virtual void allocate_memory(const int batch_size, const int prev_cols, const int prev_rows) { cout << "使用禁止i2" << endl; exit(1); }
    // getter
    virtual bool get_trainable(void) { cout << "使用禁止j" << endl; exit(1); return 1; }
    virtual string get_type(void) { cout << "使用禁止j2" << endl; exit(1); return "0"; }
    virtual int get_batch_size(void) { cout << "使用禁止k" << endl; exit(1); return 1; }
    virtual bool get_use_bias(void) { cout << "使用禁止l" << endl; exit(1); return false; }
    virtual vector<vector <MatrixXf> > get_bW(void) {
        cout << "使用禁止m" << endl;
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0].resize(1,1); tmp[0][0](0,0) = 0.f; return tmp; }
    virtual vector<vector <MatrixXf> > get_W(void) {
        cout << "使用禁止n" << endl;
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0].resize(1,1); tmp[0][0](0,0) = 0.f; return tmp; }
    virtual MatrixXf get_b(void) { cout << "使用禁止o" << endl; exit(1); return MatrixXf::Zero(1,1); }
    // virtual vector<float> get_b(void) {
    //     cout << "使用禁止o2" << endl; exit(1);
    //     vector<float> tmp{ 0.f, }; return tmp; }
    virtual int get_W_cols(void) { cout << "使用禁止p" << endl; exit(1); return 1; }
    virtual int get_W_rows(void) { cout << "使用禁止q" << endl; exit(1); return 1; }
    // virtual MatrixXf get_activated(void) { cout << "使用禁止r" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual vector< vector<MatrixXf> > get_activated(void) {
        cout << "使用禁止r2" << endl; exit(1);
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    // virtual MatrixXf get_delta(void) { cout << "使用禁止s" << endl; exit(1); return MatrixXf::Zero(1,1); }
    virtual vector<vector <MatrixXf> > get_delta(void) {
        cout << "使用禁止r2" << endl; exit(1);
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void) { cout << "使用禁止t" << endl; exit(1); return identity; }
    virtual function<MatrixXf(MatrixXf)> get_d_activateFunction(void) { cout << "使用禁止u" << endl; exit(1); return identity; }
    virtual vector<vector <MatrixXf> > get_dE_dbW(void) {
        cout << "使用禁止v" << endl;
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    virtual vector< vector<MatrixXf> > get_dE_dW(void) {
        cout << "使用禁止vv" << endl;
        vector< vector<MatrixXf> > tmp;
        tmp.resize(1); tmp[0].resize(1); tmp[0][0] = MatrixXf::Zero(1,1); return tmp; }
    virtual vector<int> get_input_map_shape(void) {cout << "使用禁止v2" << endl; exit(1); vector<int> tmp; tmp.resize(1); tmp[0] = 0; return tmp; }
    virtual vector<int> get_output_map_shape(void) {cout << "使用禁止v3" << endl; exit(1); vector<int> tmp; tmp.resize(1); tmp[0] = 0; return tmp; }
    // setter
    virtual void set_batch_size(const int batch_size) { cout << "使用禁止w" << endl; exit(1); }
    // virtual void set_batch_size(const int batch_size,
    //                             const bool use_bias_in_next_layer) { cout << "使用禁止x" << endl; exit(1); }
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
    vector<vector <MatrixXf> > _dE_dbW;
    vector<vector <MatrixXf> > bW;
    vector<vector <MatrixXf> > W;   // NOTE:convolutionしか使ってない問題
    vector< vector<MatrixXf> > dE_dW;   // NOTE:convolutionしか使ってない問題
    MatrixXf dE_db; // NOTE:convolutionしか使ってない問題
    MatrixXf b; // NOTE:convolutionしか使ってない問題
    vector<vector <MatrixXf> > _activated;
};

#endif // INCLUDE_layer_h_
