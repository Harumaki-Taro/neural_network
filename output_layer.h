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
    virtual void forwardprop(const vector<vector <MatrixXf> > X);
    virtual void calc_delta(const MatrixXf pred, const MatrixXf label);
    void build_layer(const function<MatrixXf(MatrixXf)> f,
                     const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                     const int class_num);
    virtual void allocate_memory(int);

    Output_Layer(const function<MatrixXf(MatrixXf)> f,
                 const function<float(MatrixXf, MatrixXf)> loss_func,
                 const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                 const int class_num);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual int get_batch_size(void);
    virtual vector<vector <MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);
    virtual function<MatrixXf(MatrixXf)> get_activateFunction(void);
    function<float(MatrixXf, MatrixXf)> get_lossFunction(void);
    function<MatrixXf(MatrixXf, MatrixXf)> get_deltaFunction(void);

    // setter
    virtual void set_batch_size(const int batch_size);
    virtual void set_delta(const vector<vector <MatrixXf> > delta);
    virtual void set_activateFunction(const function<MatrixXf(MatrixXf)> f);
    virtual void set_lossFunction(const function<float(MatrixXf, MatrixXf)> f);
    virtual void set_deltaFunction(const function<MatrixXf(MatrixXf, MatrixXf)> delta_f);

private:
    bool _trainable = false;
    const string type = "output_layer";
    // Parameters tracked during learning
    vector<vector <MatrixXf> > delta;
    // Parameters specified at first
    int batch_size;
    int _class_num;
    function<MatrixXf(MatrixXf)> f;
    function<float(MatrixXf, MatrixXf)> loss_func;
    function<MatrixXf(MatrixXf, MatrixXf)> delta_f;
};


Output_Layer::Output_Layer(const function<MatrixXf(MatrixXf)> f,
                           const function<float(MatrixXf, MatrixXf)> loss_func,
                           const function<MatrixXf(MatrixXf, MatrixXf)> delta_f,
                           const int class_num) {
    this->_class_num = class_num;
    this->set_activateFunction(f);
    this->loss_func = loss_func;
    this->set_deltaFunction(delta_f);
}


void Output_Layer::forwardprop(const vector<vector <MatrixXf> > X) {
    this->_activated[0][0] = this->f(X[0][0]);
}


void Output_Layer::calc_delta(const MatrixXf pred, const MatrixXf label) {
    this->delta[0][0] = this->delta_f(pred, label);
}


void Output_Layer::allocate_memory(const int batch_size) {
    this->batch_size = batch_size;

    this->delta.resize(1); this->delta[0].resize(1);
    this->delta[0][0].resize(this->_class_num, this->batch_size);

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->_class_num);
}


bool Output_Layer::get_trainable(void) { return this->_trainable; }
string Output_Layer::get_type(void) { return this->type; }
int Output_Layer::get_batch_size(void) { return this->batch_size; }
vector<vector <MatrixXf> > Output_Layer::get_activated(void) { return this->_activated; }
vector<vector <MatrixXf> > Output_Layer::get_delta(void) { return this->delta; }
function<MatrixXf(MatrixXf)> Output_Layer::get_activateFunction(void) { return this->f; }
function<float(MatrixXf, MatrixXf)> Output_Layer::get_lossFunction(void) { return this->loss_func; }
function<MatrixXf(MatrixXf, MatrixXf)> Output_Layer::get_deltaFunction(void) { return this->delta_f; }


void Output_Layer::set_batch_size(const int batch_size) {
    this->allocate_memory(batch_size);
}
void Output_Layer::set_delta(vector<vector <MatrixXf> > delta) {
    this->delta = delta;
}
void Output_Layer::set_activateFunction(const function<MatrixXf (MatrixXf)> f) {
    this->f = f;
}
void Output_Layer::set_lossFunction(function<float(MatrixXf, MatrixXf)> loss_func) {
    this->loss_func = loss_func;
}
void Output_Layer::set_deltaFunction(const function<MatrixXf (MatrixXf, MatrixXf)> delta_f) {
    this->delta_f = delta_f;
}


#endif // INCLUDE_output_layer_h_
