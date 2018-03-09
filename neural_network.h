#ifndef INCLUDE_neural_network_h_
#define INCLUDE_neural_network_h_

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <functional>
#include <math.h>
#include "Eigen/Core"
#include "layer.h"
#include "full_connect_layer.h"
#include "input_layer.h"
#include "output_layer.h"

using std::function;
using std::cout;
using std::endl;
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
    void build_fullConnectedLayer(MatrixXf, int, int,
                                 MatrixXf, int, bool,
                                 function<MatrixXf(MatrixXf)>,
                                 function<MatrixXf(MatrixXf)>);
    void build_outputLayer(int,
                           function<MatrixXf(MatrixXf)>,
                           function<MatrixXf(MatrixXf, MatrixXf)>);

    // initialize for computing
    void allocate_memory(int);

    // train or evaluate
    MatrixXf forwardprop(MatrixXf);
    void backprop(MatrixXf, MatrixXf);

    // constructor
    Neural_Network(void);

private:
    vector< shared_ptr<Layer> > layers;
    int _batch_size;
    int _example_size;
    int _label_num;
};


Neural_Network::Neural_Network(void) {
    shared_ptr<Layer> input_layer( new Input_Layer() );
    this->layers.push_back(input_layer);
}


void Neural_Network::build_fullConnectedLayer(MatrixXf W, int W_rows, int W_columns,
                                              MatrixXf b, int b_rows, bool use_bias,
                                              function<MatrixXf(MatrixXf)> f,
                                              function<MatrixXf(MatrixXf)> d_f) {

    shared_ptr<Layer> layer( new FullConnect_Layer() );
    layer->build_layer(b, W, use_bias, f, d_f);
    this->layers.push_back(layer);
}


void Neural_Network::build_outputLayer(int label_num,
                                       function<MatrixXf(MatrixXf)> f,
                                       function<MatrixXf(MatrixXf, MatrixXf)> delta_f) {
    this->_label_num = label_num;
    shared_ptr<Layer> layer( new Output_Layer() );
    layer->build_layer(this->_label_num, f, delta_f);
    this->layers.push_back(layer);
}


void Neural_Network::allocate_memory(int batch_size) {
    _batch_size = batch_size;
    auto fst_layer = ++layers.begin();
    _example_size = (*fst_layer)->get_W().rows();

    // input layer
    layers.front()->allocate_memory(_batch_size, _example_size, (*fst_layer)->get_use_bias());

    // hidden layer
    for ( int i = 1; i != (int)layers.size(); i++ ) {
        if ( i < (int)layers.size()-1 ) {
            layers[i]->allocate_memory(_batch_size, layers[i+1]->get_use_bias());
        } else {
            layers[i]->allocate_memory(_batch_size);
        }
    }
}


MatrixXf Neural_Network::forwardprop(MatrixXf X) {
    layers.front()->activated_.block(0,1,_batch_size,_example_size) = X;

    for ( int i = 1; i != (int)layers.size(); i++ ) {
        layers[i]->forwardprop(layers[i-1]->get_activated_());
    }

    return layers.back()->get_activated_();
}


void Neural_Network::backprop(MatrixXf y, MatrixXf pred) {
    layers.back()->calc_delta(y, pred);
    layers[(int)layers.size()-2]->set_delta(layers.back()->get_delta());

    for ( int i = (int)layers.size()-2; i != 1; --i ) {
        layers[i-1]->calc_delta(layers[i]->get_delta(), layers[i]->get_bW(), layers[i]->get_W_rows(), layers[i]->get_W_cols());
    }

    for ( int i = (int)layers.size()-2; i != 0; --i ) {
        layers[i]->calc_differential(layers[i-1]->get_activated_());
        layers[i]->bW = layers[i]->get_bW() + layers[i]->get_dE_dbW();
    }
}


#endif // INCLUDE_neural_network_h_
