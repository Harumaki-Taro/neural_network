#ifndef INCLUDE_neural_network_h_
#define INCLUDE_neural_network_h_

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <functional>
#include <string>
#include <math.h>
#include <float.h>
#include "Eigen/Core"
#include "layer.h"
#include "affine_layer.h"
#include "convolution_layer.h"
#include "en_tensor_layer.h"
#include "flatten_layer.h"
#include "max_pooling_layer.h"
#include "input_layer.h"
#include "output_layer.h"
#include "activate_layer.h"
#include "local_constrast_normalization_layer.h"
#include "dropout.h"
#include "my_math.h"
#include "batch.h"
#include <chrono>

using std::function;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::max;
using std::shared_ptr;
using Eigen::MatrixXf;
//
// training data
//


typedef struct tmp_layer_Wb {
    vector<vector <MatrixXf> > bW;
    vector<vector <MatrixXf> > W;
    vector<float> b;
}tmp_layer_Wb;


class Neural_Network {
    /*
    */
public:
    // build layers
    void add_layer(Affine_Layer);
    void add_layer(Output_Layer);
    void add_layer(Convolution_Layer);
    void add_layer(Max_Pooling_Layer);
    void add_layer(En_Tensor_Layer);
    void add_layer(Flatten_Layer);
    void add_layer(Activate_Layer);
    void add_layer(Dropout);
    void add_layer(LCN_Layer);

    // initialize for computing
    void allocate_memory(const int batch_size, const int example_size);

    // train or evaluate
    MatrixXf forwardprop(const MatrixXf X);
    void backprop(const MatrixXf pred, const MatrixXf label);
    float calc_loss_with_prev_pred(const MatrixXf label);
    float calc_loss(const MatrixXf X, const MatrixXf label);
    float calc_accuracy(const MatrixXf X, const MatrixXf label);
    MatrixXf inferance(const MatrixXf X);

    // debag
    void debag(const Mini_Batch mini_batch, const int point_num=3, const int calc_num_per_layer=100);
    float central_difference(const int layer_num, const int shape_0,
                             const int shape_1, const int shape_2,
                             const int shape_3, const Mini_Batch mini_batch,
                             const int point_num=3);

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
    Neural_Network(const Neural_Network &obj){ cout << "コピーコンストラクタが呼ばれたよ" << endl; } // コピーコンストラクタ
    float _logit_;
    MatrixXf _pred;
private:
    vector< shared_ptr<Layer> > _layers;
    int batch_size;
    int _example_size;
    int _class_num;
    int _layer_num;
    function<float(MatrixXf, MatrixXf)> loss_func;
};


Neural_Network::Neural_Network(void) {
    shared_ptr<Layer> input_layer( new Input_Layer() );
    this->_layers.push_back(input_layer);
}


void Neural_Network::add_layer(Affine_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Affine_Layer>(layer);
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


void Neural_Network::add_layer(Activate_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Activate_Layer>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(Dropout layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<Dropout>(layer);
    this->_layers.push_back(_layer);
}


void Neural_Network::add_layer(LCN_Layer layer) {
    std::shared_ptr<Layer> _layer = std::make_shared<LCN_Layer>(layer);
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

    // hidden layer and output layer
    for ( int i = 1; i < this->_layer_num; i++ ) {
        this->_layers[i]->allocate_memory(this->batch_size, this->_layers[i-1]);
    }
}


MatrixXf Neural_Network::forwardprop(const MatrixXf X) {
    // input layer
    this->_layers.front()->forwardprop(X);

    // hidden layer -> output layer
    for ( int i = 1; i < this->_layer_num; i++ ) {
        this->_layers[i]->forwardprop(this->_layers[i-1]->get_activated());
    }
    this->_pred = this->_layers.back()->get_activated()[0][0];
    this->_logit_ = this->_layers[this->_layers.size()-2]->get_activated()[0][0](0, 0);
    return this->_pred;
}


void Neural_Network::backprop(const MatrixXf pred, const MatrixXf label) {
    // output_layer
    this->_layers.back()->calc_delta(pred, label);
    // hidden_layer (delta)
    for ( int i = this->_layer_num-1; i != 1; --i ) {
        if ( this->_layers[i-1]->get_type() == "full_connect_layer"
            || this->_layers[i-1]->get_type() == "flatten_layer"
            || this->_layers[i-1]->get_type() == "en_tensor_layer"
            || this->_layers[i-1]->get_type() == "convolution_layer"
            || this->_layers[i-1]->get_type() == "activate_layer"
            || this->_layers[i-1]->get_type() == "dropout" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i]);
        } else if ( this->_layers[i-1]->get_type() == "max_pooling_layer"
            || this->_layers[i-1]->get_type() == "local_constract_normalization_layer" ) {
            this->_layers[i-1]->calc_delta(this->_layers[i],
                                           this->_layers[i-2]);
        } else {
            cout << i << this->_layers[i-1]->get_type() << endl;
            cout << "(calc_delta)Neural Networkクラスでは指定のレイヤークラスを利用できません。" << endl;
            exit(1);
        }
    }

    // hidden_layer (dE_db, dE_dW)
    for ( int i = this->_layer_num-1; i != 0; --i ) {
        if ( this->_layers[i]->get_trainable() ) {
            if ( this->_layers[i]->get_type() == "full_connect_layer" ) {
                this->_layers[i]->calc_differential(this->_layers[i-1],
                                                    this->_layers[i+1]);
            } else if ( this->_layers[i]->get_type() == "flatten_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "en_tensor_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "convolution_layer" ) {
                this->_layers[i]->calc_differential(this->_layers[i-1],
                                                    this->_layers[i+1]);
            } else if ( this->_layers[i]->get_type() == "max_pooling_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "activate_layer" ) {
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


float Neural_Network::calc_loss(const MatrixXf X, const MatrixXf label) {
    this->_pred = this->forwardprop(X);
    return this->calc_loss_with_prev_pred(label);
}


float Neural_Network::calc_accuracy(const MatrixXf X, const MatrixXf label) {
    float output = 0.f;
    MatrixXf answer = this->inferance(X);
    for ( int i = 0; i < this->batch_size; i++ ) {
        for ( int j = 0; j < label.cols(); j++ ) {
            if ( fabs(label(i, j) - 1.f) < 0.01 && fabs(answer(i, j) - 1.f) < 0.01 ) {
                output += 1.f;
            }
        }
    }
    return output / (float)this->batch_size;
}


MatrixXf Neural_Network::inferance(const MatrixXf X) {
    this->forwardprop(X);
    MatrixXf output = MatrixXf::Zero(this->_pred.rows(), this->_pred.cols());
    for ( int i = 0; i < this->_pred.rows(); ++i ) {
        float tmp = 0.f;
        int n = 0;
        for ( int j = 0; j < this->_pred.cols(); ++j ) {
            if ( tmp < this->_pred(i, j) ) {
                tmp = this->_pred(i, j);
                n = j;
            }
        }
        output(i, n) = 1.f;
    }

    return output;
}


void Neural_Network::debag(const Mini_Batch mini_batch, const int point_num, const int calc_num_per_layer) {
    // Automatic differentiation with back propagation
    MatrixXf pred = this->forwardprop(mini_batch.example);
    this->backprop(pred, mini_batch.label);
    float loss_value = this->calc_loss_with_prev_pred(mini_batch.label);

    // Numerical differentiation with central difference formula
    for ( int i = 0; i != (int)this->_layers.size(); i++ ) {
        if ( this->_layers[i]->get_trainable() ) {
            if ( this->_layers[i]->get_type() == "full_connect_layer" ) {
                vector<float> aut_diff;
                vector<float> nmc_diff;
                float mean_diff_diff=0.f;
                vector<int> shape_0 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W.size()-1);
                vector<int> shape_1 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0].size()-1);
                vector<int> shape_2 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0][0].rows()-1);
                vector<int> shape_3 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0][0].cols()-1);

                for ( int j = 0; j < calc_num_per_layer; j++ ) {
                    nmc_diff.push_back(this->central_difference(i, shape_0[j], shape_1[j], shape_2[j], shape_3[j],
                                                                mini_batch, point_num));
                    aut_diff.push_back(this->_layers[i]->dE_dW[shape_0[j]][shape_1[j]](shape_2[j], shape_3[j]));
                    mean_diff_diff += fabs(nmc_diff[j] - aut_diff[j]);
                }
                mean_diff_diff /= (float)calc_num_per_layer;
                cout << "Layer" << i << " (full_connect_layer): " << mean_diff_diff / loss_value * 100.f << "[\%]" << endl;

            } else if ( this->_layers[i]->get_type() == "flatten_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "en_tensor_layer" ) {
                ;
            } else if ( this->_layers[i]->get_type() == "convolution_layer" ) {
                vector<float> aut_diff;
                vector<float> nmc_diff;
                float mean_diff_diff=0.f;
                vector<int> shape_0 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W.size()-1);
                vector<int> shape_1 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0].size()-1);
                vector<int> shape_2 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0][0].rows()-1);
                vector<int> shape_3 = rand_array(calc_num_per_layer, 0, this->_layers[i]->W[0][0].cols()-1);

                for ( int j = 0; j < calc_num_per_layer; j++ ) {
                    nmc_diff.push_back(this->central_difference(i, shape_0[j], shape_1[j], shape_2[j], shape_3[j],
                                                                mini_batch, point_num));
                    aut_diff.push_back(this->_layers[i]->dE_dW[shape_0[j]][shape_1[j]](shape_2[j], shape_3[j]));
                    mean_diff_diff += fabs(nmc_diff[j] - aut_diff[j]);
                }
                mean_diff_diff /= (float)calc_num_per_layer;
                cout << "Layer" << i << " (convolution_layer): " << mean_diff_diff / loss_value * 100.f << "[\%]" << endl;
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


float Neural_Network::central_difference(const int layer_num, const int shape_0,
                                         const int shape_1, const int shape_2,
                                         const int shape_3, const Mini_Batch mini_batch,
                                         const int point_num) {

    float tmp = this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3);
    float eps;
    float _eps = 0.02;
    if ( fabs(tmp) * _eps > FLT_EPSILON ) {
        eps = fabs(tmp) * _eps;
    } else {
        eps = FLT_EPSILON;
    }
    float output;

    if ( point_num == 3 ) {
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp - eps;
        float left = this->calc_loss(mini_batch.example, mini_batch.label);
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp + eps;
        float right = this->calc_loss(mini_batch.example, mini_batch.label);

        output = (right - left) / (2.f * eps);
    } else if ( point_num == 5 ) {
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp - 2.f * eps;
        float left_left = this->calc_loss(mini_batch.example, mini_batch.label);
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp - eps;
        float left = this->calc_loss(mini_batch.example, mini_batch.label);
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp + eps;
        float right = this->calc_loss(mini_batch.example, mini_batch.label);
        this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp + 2.f * eps;
        float right_right = this->calc_loss(mini_batch.example, mini_batch.label);

        output = (left_left - 8.f * left + 8.f * right - right_right) / (12.f * eps);
    } else {
        cout << "対応していません" << endl;
        exit(1);
    }
    this->_layers[layer_num]->W[shape_0][shape_1](shape_2, shape_3) = tmp;

    return output;
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
