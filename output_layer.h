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
    virtual void forwardprop(const MatrixXf X);
    virtual void calc_delta(const MatrixXf label, const MatrixXf pred);
    virtual void build_layer(const function<MatrixXf(MatrixXf)> f,
                             const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                             const int class_num);
    virtual void allocate_memory(int);

    // getter
    virtual bool get_trainable(void);
    virtual int get_batch_size(void);
    virtual MatrixXf get_activated(void);
    virtual MatrixXf get_delta(void);
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void);
    function<MatrixXf(MatrixXf, MatrixXf)> get_deltaFunction(void);

    // setter
    virtual void set_batch_size(const int batch_size);
    void set_delta(const MatrixXf delta);
    void set_activateFunction(const function<MatrixXf(MatrixXf)> f);
    void set_deltaFunction(const function<MatrixXf(MatrixXf, MatrixXf)> delta_f);

private:
    bool _trainable = false;
    // Parameters tracked during learning
    MatrixXf delta;
    // Parameters specified at first
    int batch_size;
    int _class_num;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf, MatrixXf)> delta_f;
};


void Output_Layer::forwardprop(const MatrixXf X) {
    this->_activated = f(X.block(0,1,X.rows(),X.cols()-1));
}


void Output_Layer::calc_delta(const MatrixXf label, const MatrixXf pred) {
    this->delta = delta_f(label, pred);
}


void Output_Layer::build_layer(const function<MatrixXf(MatrixXf)> f,
                               const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                               const int class_num) {
    this->_class_num = class_num;
    set_activateFunction(f);
    set_deltaFunction(delta_f);
    }


void Output_Layer::allocate_memory(const int batch_size) {
    this->batch_size = batch_size;
    this->delta.resize(this->batch_size, this->_class_num);

    this->_activated.resize(this->batch_size, this->_class_num);
}


bool Output_Layer::get_trainable(void) { return this->_trainable; }
int Output_Layer::get_batch_size(void) { return this->batch_size; }
MatrixXf Output_Layer::get_activated(void) { return this->_activated; }
MatrixXf Output_Layer::get_delta(void) { return this->delta; }
function<MatrixXf(MatrixXf)> Output_Layer::get_activateFunction(void) { return this->f; }
function<MatrixXf(MatrixXf, MatrixXf)> Output_Layer::get_deltaFunction(void) { return this->delta_f; }


void Output_Layer::set_batch_size(const int batch_size) {
    this->allocate_memory(batch_size);
}
void Output_Layer::set_delta(MatrixXf delta) {
    this->delta = delta;
}
void Output_Layer::set_activateFunction(const function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void Output_Layer::set_deltaFunction(const function<MatrixXf (MatrixXf, MatrixXf)> delta_f) {
    this->delta_f = delta_f;
}


#endif // INCLUDE_output_layer_h_
