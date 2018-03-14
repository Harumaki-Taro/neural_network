#ifndef INCLUDE_neural_network_h_
#define INCLUDE_neural_network_h_

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <functional>
#include <string>
#include <math.h>
#include "Eigen/Core"
#include "layer.h"
#include "full_connect_layer.h"
#include "input_layer.h"
#include "output_layer.h"
#include "my_math.h"

using std::function;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using Eigen::MatrixXf;
//
// training data
//


class Neural_Network {
    /*
    */
public:
    // build layers
    void build_fullConnectedLayer(const function<MatrixXf(MatrixXf)> f,
                                  const function<MatrixXf(MatrixXf)> d_f,
                                  const MatrixXf W, const MatrixXf b,
                                  const bool use_bias);
    void build_fullConnectedLayer(const function<MatrixXf(MatrixXf)> f,
                                  const function<MatrixXf(MatrixXf)> d_f,
                                  const int (&W_shape)[2], const bool use_bias,
                                  const float W_min=-1.f, const float W_max=1.f,
                                  const float b_min=-1.f, const float b_max=1.f);
    // void build_convolutionLayer(const function<MatrixXf(MatrixXf)> f,
    //                             const function<MatrixXf(MatrixXf)> d_f,
    //                             const int prev_ch, const int ch,
    //                             const int filter_height, const int filter_width,
    //                             const int stlide_height, const int stlide_width,
    //                             const int padding_height, const int padding_width,
    //                             const float W_min, const float W_max,
    //                             const float b_min, const float b_max)
    void build_outputLayer(const int class_num,
                           const function<MatrixXf(MatrixXf)> f,
                           const string loss_name);

    // initialize for computing
    void allocate_memory(const int batch_size);

    // train or evaluate
    MatrixXf forwardprop(const MatrixXf X);
    void backprop(const MatrixXf pred, const MatrixXf label);
    float calc_loss_with_prev_pred(const MatrixXf label);
    float calc_loss(const MatrixXf X, const MatrixXf label);

    // getter
    vector< shared_ptr<Layer> > get_layers(void);
    int get_batch_size(void);
    int get_example_size(void);
    int get_class_num(void);
    int get_layer_num(void);
    MatrixXf get_pred(void);
    function<float(MatrixXf, MatrixXf)> get_loss_func(void);

    // setter
    void set_batch_size(const int batch_size);
    void set_loss_func(const function<float(MatrixXf, MatrixXf)> loss_name);

    // constructor
    Neural_Network(void);

private:
    vector< shared_ptr<Layer> > _layers;
    int batch_size;
    int _example_size;
    int _class_num;
    int _layer_num;
    MatrixXf _pred;
    function<float(MatrixXf, MatrixXf)> loss_func;
};


Neural_Network::Neural_Network(void) {
    shared_ptr<Layer> input_layer( new Input_Layer() );
    this->_layers.push_back(input_layer);
}


void Neural_Network::build_fullConnectedLayer(const function<MatrixXf(MatrixXf)> f,
                                              const function<MatrixXf(MatrixXf)> d_f,
                                              const MatrixXf W, const MatrixXf b,
                                              const bool use_bias) {

    shared_ptr<Layer> layer( new FullConnect_Layer() );
    layer->build_layer(f, d_f, W, b, use_bias);
    this->_layers.push_back(layer);
}


void Neural_Network::build_fullConnectedLayer(const function<MatrixXf(MatrixXf)> f,
                                              const function<MatrixXf(MatrixXf)> d_f,
                                              const int (&W_shape)[2], const bool use_bias,
                                              const float W_min, const float W_max,
                                              const float b_min, const float b_max) {

    shared_ptr<Layer> layer( new FullConnect_Layer() );
    layer->build_layer(f, d_f, W_shape, use_bias, W_min, W_max, b_min, b_max);
    this->_layers.push_back(layer);
}


// void Neural_Network::build_convolutionLayer(const function<MatrixXf(MatrixXf)> f,
//                                             const function<MatrixXf(MatrixXf)> d_f,
//                                             const int prev_ch, const int ch,
//                                             const int filter_height, const int filter_width,
//                                             const int stlide_height, const int stlide_width,
//                                             const int padding_height, const int padding_width,
//                                             const float W_min, const float W_max,
//                                             const float b_min, const float b_max) {
//
//     shared_ptr<Layer> layer( new FullConnect_Layer() );
//     layer->build_layer(f, d_f, W_shape, use_bias, W_min, W_max, b_min, b_max);
//     this->_layers.push_back(layer);
// }


void Neural_Network::build_outputLayer(const int class_num,
                                       const function<MatrixXf(MatrixXf)> f,
                                       const string loss_name) {
    this->_class_num = class_num;
    shared_ptr<Layer> output_layer( new Output_Layer() );
    if ( loss_name == "mean_square_error" ) {
        output_layer->build_layer(f, diff, this->_class_num);
        this->loss_func = mean_square_error;
    } else if ( loss_name == "mean_cross_entropy" ) {
        output_layer->build_layer(f, diff, this->_class_num);
        this->loss_func = mean_cross_entropy;
    } else {
        cout << "現在、指定の損失関数はOutput_layerクラスでは利用できません。" << endl;
        exit(1);
    }
    this->_layers.push_back(output_layer);
}


void Neural_Network::allocate_memory(const int batch_size) {
    this->batch_size = batch_size;
    this->_layer_num = this->_layers.size();
    auto fst_layer = ++this->_layers.begin();
    this->_example_size = (*fst_layer)->get_W()[0][0].rows();

    // input layer
    this->_layers.front()->allocate_memory(this->batch_size,
                                           this->_example_size,
                                           (*fst_layer)->get_use_bias());

    // hidden layer
    for ( int i = 1; i < this->_layer_num-1; i++ ) {
        if ( i < this->_layer_num-2 ) {
            this->_layers[i]->allocate_memory(this->batch_size,
                                              this->_layers[i+1]->get_use_bias());
        } else {
            this->_layers[i]->allocate_memory(this->batch_size);
        }
    }

    // output_layer
    this->_layers.back()->allocate_memory(this->batch_size);
}


MatrixXf Neural_Network::forwardprop(const MatrixXf X) {
    // input layer
    this->_layers.front()->_activated[0][0].block(0,1,this->batch_size,this->_example_size) = X;

    // hidden layer -> output layer
    for ( int i = 1; i < this->_layer_num; i++ ) {
        this->_layers[i]->forwardprop(this->_layers[i-1]->get_activated());
    }
    this->_pred = this->_layers.back()->get_activated()[0][0];

    return this->_pred;
}


void Neural_Network::backprop(const MatrixXf pred, const MatrixXf label) {
    // output_layer
    this->_layers.back()->calc_delta(pred, label);
    this->_layers[this->_layer_num-2]->set_delta(this->_layers.back()->get_delta());

    // hidden_layer
    for ( int i = this->_layer_num-2; i != 1; --i ) {
        this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta(),
                                       this->_layers[i]->get_bW(),
                                       this->_layers[i]->get_W_rows(),
                                       this->_layers[i]->get_W_cols());
    }
}


float Neural_Network::calc_loss_with_prev_pred(const MatrixXf label) {
    return this->loss_func(this->_pred, label);
}


float Neural_Network::calc_loss(MatrixXf X, MatrixXf label) {
    this->_pred = this->forwardprop(X);
    return this->calc_loss_with_prev_pred(label);
}


vector< shared_ptr<Layer> > Neural_Network::get_layers(void) {
    return this->_layers;
}
int Neural_Network::get_batch_size(void) {
    return this->batch_size;
}
int Neural_Network::get_example_size(void) {
    return this->_example_size;
}
int Neural_Network::get_class_num(void) {
    return this->_class_num;
}
int Neural_Network::get_layer_num(void) {
    return this->_layer_num;
}
MatrixXf Neural_Network::get_pred(void) {
    return this->_pred;
}
function<float(MatrixXf, MatrixXf)> Neural_Network::get_loss_func(void) {
    return this->loss_func;
}


void Neural_Network::set_batch_size(const int batch_size) {
    /*
        Chane batch size and re-allocate memory.
        <NONE>
            All except weight and bias are initialized.
    */
    this->allocate_memory(batch_size);
}
void Neural_Network::set_loss_func(const function<float(MatrixXf, MatrixXf)> loss_func) {
    this->loss_func = loss_func;
}

#endif // INCLUDE_neural_network_h_
