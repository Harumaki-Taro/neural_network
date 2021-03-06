#ifndef INCLUDE_en_tensor_layer_h_
#define INCLUDE_en_tensor_layer_h_

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
using std::make_shared;
using Eigen::MatrixXf;


class En_Tensor_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer);

    En_Tensor_Layer(const int channel_num, const int height, const int width);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<int> get_output_map_shape(void);

private:
    bool trainable = false;
    const string type = "en_tensor_layer";
    const bool is_tensor = true;
    int unit_num;
    int channel_num;
    int output_height;
    int output_width;
    int batch_size;
};


En_Tensor_Layer::En_Tensor_Layer(const int channel_num, const int height, const int width) {
    this->channel_num = channel_num;
    this->output_height = height;
    this->output_width = width;
}


void En_Tensor_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int k = 0; k < this->channel_num; k++ ) {
            for ( int h = 0; h < this->output_height; h++ ) {
                for ( int w = 0; w < this->output_width; w++ ) {
                    this->_activated[n][k](h, w)
                        = X[0][0](n, this->output_height * this->output_width * k + this->output_width * h + w);
                }
            }
        }
    }
}


void En_Tensor_Layer::calc_delta(const shared_ptr<Layer> &next_layer) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        shared_ptr<MatrixXf> next_delta;
        for ( int k = 0; k <this->channel_num; k++ ) {
            next_delta = make_shared<MatrixXf>(next_layer->delta[n][k]);
            for ( int p = 0; p < this->output_height; p++ ) {
                for ( int q = 0; q < this->output_width; q++ ) {
                    this->delta[0][0](n, this->output_height * this->output_width * k + this->output_width * p + q)
                        =  (*next_delta)(p, q);
                }
            }
        }
    }
}


void En_Tensor_Layer::allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->unit_num = this->channel_num * this->output_height * this->output_width;
    this->_activated.resize(this->batch_size);
    for ( int n = 0; n < this->batch_size; n++ ) {
        this->_activated[n].resize(this->channel_num);
        for ( int k = 0; k < this->channel_num; k++ ) {
            this->_activated[n][k].resize(this->output_height, this->output_width);
        }
    }

    this->delta.resize(1); this->delta[0].resize(1);
    this->delta[0][0].resize(this->batch_size, this->channel_num*this->output_height*this->output_width);
}


bool En_Tensor_Layer::get_trainable(void) { return this->trainable; }
string En_Tensor_Layer::get_type(void) { return this->type; }
bool En_Tensor_Layer::get_is_tensor(void) { return this->is_tensor; }
int En_Tensor_Layer::get_unit_num(void) { return this->unit_num; }
vector< vector<MatrixXf> > En_Tensor_Layer::get_activated(void) {
    return this->_activated;
}
vector<int> En_Tensor_Layer::get_output_map_shape(void) {
    vector<int> output_map_shape{ this->output_height, this->output_width };
    return output_map_shape;
}


#endif // INCLUDE_en_tensor_layer_h_
