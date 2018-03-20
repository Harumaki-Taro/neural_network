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


class Tensor_Activate_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const int height, const int width);

    Tensor_Activate_Layer(const function<MatrixXf(MatrixXf)> f,
                          const function<MatrixXf(MatrixXf)> d_f,
                          const int ch);

    virtual vector<int> get_input_map_shape(void);
    virtual vector<int> get_output_map_shape(void);
    virtual string get_type(void);
    virtual vector< vector<MatrixXf> > get_delta(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual bool get_trainable(void);

private:
    string type = "tensor_activate_layer";
    bool trainable = false;
    int batch_size;
    int channel_num;
    int height;
    int width;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
};


Tensor_Activate_Layer::Tensor_Activate_Layer(const function<MatrixXf(MatrixXf)> f,
                                             const function<MatrixXf(MatrixXf)> d_f,
                                             const int ch) {
    this->f = f;
    this->d_f = d_f;
    this->channel_num = ch;
}


void Tensor_Activate_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->_activated[n][c] = this->f(X[n][c]);
        }
    }
}


void Tensor_Activate_Layer::calc_delta(const shared_ptr<Layer> &next_layer) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->delta[n][c] = ((next_layer->get_delta()[n][c]).array() * (this->d_f(this->_activated[n][c])).array()).matrix();
        }
    }
}


void Tensor_Activate_Layer::allocate_memory(const int batch_size, const int height, const int width) {
    this->batch_size = batch_size;
    this->height = height;
    this->width = width;

    // preActivate & activated
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->height, this->width));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->height, this->width));
        }
        this->delta.push_back(tmp_delta);
    }
}


vector<int> Tensor_Activate_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape{ this->height, this->width };
    return input_map_shape;
}
vector<int> Tensor_Activate_Layer::get_output_map_shape(void) {
    vector<int> output_map_shape{ this->height, this->width };
    return output_map_shape;
}
string Tensor_Activate_Layer::get_type(void) {
    return this->type;
}
vector<vector <MatrixXf> > Tensor_Activate_Layer::get_delta(void) {
    return this->delta;
}
vector<vector <MatrixXf> > Tensor_Activate_Layer::get_activated(void) {
    return this->_activated;
}
bool Tensor_Activate_Layer::get_trainable(void) {
    return this->trainable;
}


#endif // INCLUDE_tensor_activate_layer_h_
