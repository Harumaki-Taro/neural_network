#ifndef INCLUDE_output_layer_h_
#define INCLUDE_output_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class Output_Layer : public Layer {
public:
    virtual void forwardprop(MatrixXf);
    virtual void calc_delta(MatrixXf, MatrixXf);
    virtual void build_layer(int,
                             function<MatrixXf(MatrixXf)>,
                             function<MatrixXf(MatrixXf, MatrixXf)>);
    virtual void allocate_memory(int);
    // setter
    void set_activateFunction(function<MatrixXf(MatrixXf)>);
    void set_deltaFunction(function<MatrixXf(MatrixXf, MatrixXf)>);
    void set_delta(MatrixXf);
    // getter
    function<MatrixXf(MatrixXf)> get_f(void);
    function<MatrixXf(MatrixXf, MatrixXf)> get_delta_f(void);
    virtual MatrixXf get_activated_(void);
    virtual MatrixXf get_delta(void);
    int get_batch_size(void);
    bool get_use_bias(void);
    virtual bool get_trainable(void);

private:
    bool trainable = false;
    // Parameters tracked during learning
    MatrixXf delta;
    // Parameters specified at first
    int batch_size;
    int label_num;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf, MatrixXf)> delta_f;
};


void Output_Layer::forwardprop(MatrixXf X) {
    activated_ = f(X.block(0,1,X.rows(),X.cols()-1));
}


void Output_Layer::calc_delta(MatrixXf y, MatrixXf pred) {
    this->delta = delta_f(y, pred);
}


void Output_Layer::build_layer(int label_num,
                               function<MatrixXf(MatrixXf)> f,
                               function<MatrixXf(MatrixXf, MatrixXf)> delta_f) {
    this->label_num = label_num;
    set_activateFunction(f);
    set_deltaFunction(delta_f);
    }


void Output_Layer::allocate_memory(int batch_size) {
    this->batch_size = batch_size;
    delta.resize(this->batch_size, this->label_num);

    activated_.resize(this->batch_size, this->label_num);
}


void Output_Layer::set_activateFunction(function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void Output_Layer::set_deltaFunction(function<MatrixXf (MatrixXf, MatrixXf)> delta_f) {
    this->delta_f = delta_f;
}
void Output_Layer::set_delta(MatrixXf delta) {
    this->delta = delta;
}


function<MatrixXf(MatrixXf)> Output_Layer::get_f(void) { return this->f; }
function<MatrixXf(MatrixXf, MatrixXf)> Output_Layer::get_delta_f(void) { return this->delta_f; }
MatrixXf Output_Layer::get_activated_(void) { return this->activated_; }
MatrixXf Output_Layer::get_delta(void) { return this->delta; }
int Output_Layer::get_batch_size(void) { return this->batch_size; }
bool Output_Layer::get_use_bias(void) { return false; }
bool Output_Layer::get_trainable(void) { return this->trainable; }

#endif // INCLUDE_output_layer_h_
