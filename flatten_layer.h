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
using std::shared_ptr;
using Eigen::MatrixXf;


class Flatten_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > next_delta);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer);

    Flatten_Layer(const int channel_num, const int height, const int width);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<int> get_input_map_shape(void);
    virtual vector< vector<MatrixXf> > get_delta(void);


private:
    bool trainable = false;
    const string type = "flatten_layer";
    const bool is_tensor = false;
    int unit_num;
    int prev_channel_num;
    int input_height;
    int input_width;
    int batch_size;
};


Flatten_Layer::Flatten_Layer(const int channel_num, const int height, const int width) {
    this->prev_channel_num = channel_num;
    this->input_height = height;
    this->input_width = width;
}


void Flatten_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int c = 0; c < this->prev_channel_num; c++ ) {
            for ( int h = 0; h < this->input_height; h++ ) {
                for ( int w = 0; w < this->input_width; w++ ) {
                    this->_activated[0][0](n, this->input_height * this->input_width * c + this->input_width * h + w)
                        = X[n][c](h, w);
                }
            }
        }
    }
}


void Flatten_Layer::calc_delta(const shared_ptr<Layer> &next_layer) {
    vector< vector<MatrixXf> > tmp;
    tmp.resize(1); tmp[0].resize(1);
    tmp[0][0].resize(this->batch_size, this->prev_channel_num * this->input_height * this->input_width);
    tmp[0][0] = next_layer->get_delta()[0][0]
        * next_layer->W[0][0].block(1,0,next_layer->get_W_rows(),next_layer->get_W_cols()).transpose();

    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int c = 0; c < this->prev_channel_num; c++ ) {
            for ( int h = 0; h < this->input_height; h++ ) {
                for ( int w = 0; w < this->input_width; w++ ) {
                    this->delta[n][c](h, w)
                        = tmp[0][0](n, this->input_height * this->input_width * c + this->input_width * h + w);
                }
            }
        }
    }
}


void Flatten_Layer::allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->unit_num = this->prev_channel_num * this->input_height * this->input_width;

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->prev_channel_num*this->input_height*this->input_width);

    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->prev_channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->input_height, this->input_width));
        }
        this->delta.push_back(tmp_delta);
    }
}


bool Flatten_Layer::get_trainable(void) { return this->trainable; }
string Flatten_Layer::get_type(void) { return this->type; }
bool Flatten_Layer::get_is_tensor(void) { return this->is_tensor; }
int Flatten_Layer::get_unit_num(void) { return this->unit_num; }
vector< vector<MatrixXf> > Flatten_Layer::get_activated(void) { return this->_activated; }
vector<int> Flatten_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape{ this->input_height, this->input_width };
    return input_map_shape;
}
vector< vector<MatrixXf> > Flatten_Layer::get_delta(void) { return this->delta; }


#endif // INCLUDE_flatten_layer_h_
