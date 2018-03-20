#ifndef INCLUDE_tensor_activate_layer_h_
#define INCLUDE_tensor_activate_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"
#include <chrono>

using std::function;
using std::cout;
using std::endl;
using std::make_shared;
using std::max;
using std::min;
using std::shared_ptr;
using Eigen::MatrixXf;


class Activate_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer);

    Activate_Layer(const function<MatrixXf(MatrixXf)> f,
                          const function<MatrixXf(MatrixXf)> d_f,
                          const int ch);

    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual vector<int> get_input_map_shape(void);
    virtual vector<int> get_output_map_shape(void);
    virtual vector< vector<MatrixXf> > get_delta(void);
    virtual vector< vector<MatrixXf> > get_activated(void);

private:
    string type = "activate_layer";
    bool trainable = false;
    bool is_tensor;
    int batch_size;
    int unit_num;
    vector<int> X_shape;
    int channel_num;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
};


Activate_Layer::Activate_Layer(const function<MatrixXf(MatrixXf)> f,
                               const function<MatrixXf(MatrixXf)> d_f,
                               const int ch) {
    this->f = f;
    this->d_f = d_f;
}


void Activate_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->X_shape[0]; ++n ) {
        for ( int c = 0; c < this->X_shape[1]; ++c ) {
            this->_activated[n][c] = this->f(X[n][c]);
        }
    }
}


void Activate_Layer::calc_delta(const shared_ptr<Layer> &next_layer) {
    #pragma omp parallel for
    for ( int n = 0; n < this->X_shape[0]; ++n ) {
        for ( int c = 0; c < this->X_shape[1]; ++c ) {
            this->delta[n][c] = ((next_layer->get_delta()[n][c]).array() * (this->d_f(this->_activated[n][c])).array()).matrix();
        }
    }
}


void Activate_Layer::allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    // this->height = height;
    // this->width = width;
    this->X_shape.resize(4);
    this->is_tensor = prev_layer->get_is_tensor();
    this->channel_num = prev_layer->get_channel_num();
    if ( this->is_tensor ) {
        this->X_shape[0] = this->batch_size;
        this->X_shape[1] = prev_layer->get_channel_num();
        this->X_shape[2] = prev_layer->get_output_map_shape()[0];
        this->X_shape[3] = prev_layer->get_output_map_shape()[1];
    } else {
        this->X_shape[0] = 1;
        this->X_shape[1] = prev_layer->get_channel_num();
        this->X_shape[2] = this->batch_size;
        this->X_shape[3] = prev_layer->get_unit_num();
    }
    this->unit_num = this->X_shape[0] * this->X_shape[1] * this->X_shape[2] * this->X_shape[3];

    // preActivate & activated
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->delta.push_back(tmp_delta);
    }
}

bool Activate_Layer::get_trainable(void) { return this->trainable; }
string Activate_Layer::get_type(void) { return this->type; }
bool Activate_Layer::get_is_tensor(void) { return this->is_tensor; }
int Activate_Layer::get_unit_num(void) { return this->unit_num; }
vector<int> Activate_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape;
    if ( this->is_tensor ) {
        input_map_shape = { this->X_shape[2], this->X_shape[3] };
    } else {
        cout << "この活性化層の直前の層はtensorではありません" << endl;
        exit(1);
    }

    return input_map_shape;
}
vector<int> Activate_Layer::get_output_map_shape(void) {
    return this->get_input_map_shape();
}
vector<vector <MatrixXf> > Activate_Layer::get_delta(void) {
    return this->delta;
}
vector<vector <MatrixXf> > Activate_Layer::get_activated(void) {
    return this->_activated;
}


#endif // INCLUDE_tensor_activate_layer_h_
