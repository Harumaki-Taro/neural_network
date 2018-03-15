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
#include "convolution_layer.h"
#include "en_tensor_layer.h"
#include "flatten_layer.h"
#include "max_pooling_layer.h"
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
    void add_layer(FullConnect_Layer);
    void add_layer(Output_Layer);
    void add_layer(Convolution_Layer);
    void add_layer(Max_Pooling_Layer);
    void add_layer(En_Tensor_Layer);
    void add_layer(Flatten_Layer);

    // initialize for computing
    void allocate_memory(const int batch_size, const int example_size);

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


void Neural_Network::add_layer(FullConnect_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<FullConnect_Layer>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(Output_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Output_Layer>(layer);
    this->_layers.push_back(_layer);
    this->loss_func = layer.get_lossFunction();
}


void Neural_Network::add_layer(Convolution_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Convolution_Layer>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(Max_Pooling_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Max_Pooling_Layer>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(En_Tensor_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<En_Tensor_Layer>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(Flatten_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Flatten_Layer>(layer);
    this->_layers.push_back(_layer);
}


//NOTE: 本当はバッチサイズだけで良いはずなのでどうにかする問題
void Neural_Network::allocate_memory(const int batch_size, const int example_size) {
    this->batch_size = batch_size;
    this->_layer_num = this->_layers.size();
    this->_example_size = example_size;

    // input layer
    this->_layers.front()->allocate_memory(this->batch_size,
                                           this->_example_size);

    // hidden layer
    for ( int i = 1; i < this->_layer_num-1; i++ ) {
        if ( this->_layers[i]->get_type() == "full_connect_layer" ) {
            this->_layers[i]->allocate_memory(this->batch_size);
        } else if ( this->_layers[i]->get_type() == "en_tensor_layer" ) {
            this->_layers[i]->allocate_memory(this->batch_size);
        } else if ( this->_layers[i]->get_type() == "flatten_layer" ) {
            this->_layers[i]->allocate_memory(this->batch_size);
        } else if ( this->_layers[i]->get_type() == "convolution_layer" ) {
            this->_layers[i]->allocate_memory(this->batch_size,
                                              this->_layers[i-1]->get_output_map_shape()[0],
                                              this->_layers[i-1]->get_output_map_shape()[1]);
        } else if ( this->_layers[i]->get_type() == "max_pooling_layer" ) {
            this->_layers[i]->allocate_memory(this->batch_size,
                                             this->_layers[i-1]->get_output_map_shape()[0],
                                             this->_layers[i-1]->get_output_map_shape()[1]);
        } else {
            cout << this->_layers[i]->get_type() << endl;
            cout << "Neural Networkクラスでは指定のレイヤークラスを利用できません。" << endl;
            exit(1);
        }
    }
    // output_layer
    this->_layers.back()->allocate_memory(this->batch_size);
}


MatrixXf Neural_Network::forwardprop(const MatrixXf X) {
    // input layer
    this->_layers.front()->forwardprop(X);

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
    // hidden_layer (delta)
    for ( int i = this->_layer_num-2; i != 1; --i ) {
        if ( this->_layers[i-1]->get_type() == "full_connect_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta(),
                                           this->_layers[i]->get_bW(),
                                           this->_layers[i]->get_W_rows(),
                                           this->_layers[i]->get_W_cols());
        } else if ( this->_layers[i-1]->get_type() == "flatten_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta(),
                                           this->_layers[i]->get_bW(),
                                           this->_layers[i]->get_W_rows(),
                                           this->_layers[i]->get_W_cols());
        } else if ( this->_layers[i-1]->get_type() == "en_tensor_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta());
        } else if ( this->_layers[i-1]->get_type() == "convolution_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta());
        } else if ( this->_layers[i-1]->get_type() == "max_pooling_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]->get_delta(),
                                           this->_layers[i-2]->get_activated());
        } else {
            cout << i << this->_layers[i-1]->get_type() << endl;
            cout << "(calc_delta)Neural Networkクラスでは指定のレイヤークラスを利用できません。" << endl;
            exit(1);
        }
    }

    // hidden_layer (dE_db, dE_dW)
    for ( int i = 0; i != (int)this->_layers.size(); i++ ) {
        if ( this->_layers[i]->get_trainable() ) {
            if ( this->_layers[i]->get_type() == "full_connect_layer" ) {
                this->_layers[i]->calc_differential(this->_layers[i-1]->get_activated());
            } else if ( this->_layers[i]->get_type() == "flatten_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "en_tensor_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "convolution_layer" ) {
                this->_layers[i]->calc_differential(this->_layers[i-1]->get_activated(),
                                                    this->_layers[i+1]->get_delta());
            } else if ( this->_layers[i]->get_type() == "max_pooling_layer" ) {
                ;
            } else {
                cout << i << this->_layers[i]->get_type() << endl;
                cout << "(calc_dE)Neural Networkクラスでは指定のレイヤークラスを利用できません。" << endl;
                exit(1);
            }
        }
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
    this->allocate_memory(batch_size, this->_example_size);
}
void Neural_Network::set_loss_func(const function<float(MatrixXf, MatrixXf)> loss_func) {
    this->loss_func = loss_func;
}

#endif // INCLUDE_neural_network_h_
