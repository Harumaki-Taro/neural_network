#ifndef INCLUDE_flatten_layer_h_
#define INCLUDE_flatten_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;
using std::shared_ptr;


class Flatten_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > next_delta);
    virtual void calc_delta(const MatrixXf next_delta);
    virtual void build_layer(const int channel_num, const int height, const int width);
    virtual void allocate_memory(const int batch_size);

    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<int> get_input_map_shape(void);

private:
    bool trainable = false;
    const string type = "flatten_layer";
    int prev_channel_num;
    int input_height;
    int input_width;
    int batch_size;
    vector< vector<MatrixXf> > delta;
};


void Flatten_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int c = 0; c < this->prev_channel_num; c++ ) {
            for ( int h = 0; h < this->input_height; h++ ) {
                for ( int w = 0; w < this->input_width; w++ ) {
                    this->_activated[0][0](n, this->prev_channel_num * this->input_height * c + this->input_height * h + w)
                        = X[n][c](h,w);
                }
            }
        }
    }
}


void Flatten_Layer::calc_delta(const MatrixXf next_delta) {
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int c = 0; this->prev_channel_num; c++ ) {
            for ( int h = 0; this->input_height; h++ ) {
                for ( int w = 0; this->input_width; w++ ) {
                    this->delta[n][c](h, w) = next_delta(n, this->prev_channel_num * this->input_height * h + w);
                }
            }
        }
    }
}


void Flatten_Layer::build_layer(const int channel_num, const int height, const int width) {
    this->prev_channel_num = channel_num;
    this->input_height = height;
    this->input_width = width;
}


void Flatten_Layer::allocate_memory(const int batch_size) {
    this->batch_size = batch_size;
    this->_activated.resize(0); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->prev_channel_num*this->input_height*this->input_width);
}


bool Flatten_Layer::get_trainable(void) { return this->trainable; }
string Flatten_Layer::get_type(void) { return this->type; }
vector< vector<MatrixXf> > Flatten_Layer::get_activated(void) { return this->_activated; }
vector<int> Flatten_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape{ this->input_height, this->input_width };
    return input_map_shape;
}


#endif // INCLUDE_flatten_layer_h_
