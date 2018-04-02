#ifndef INCLUDE_local_response_normalization_layer_h_
#define INCLUDE_local_response_normalization_layer_h_

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
using std::max;
using std::min;
using Eigen::MatrixXf;

// NOTE:本当に正しいか要チェック
class LRN_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer,
                            const shared_ptr<Layer> &prev_layer);
    virtual void allocate_memory(const int batch_size,
                                 const shared_ptr<Layer> &prev_layer);

    LRN_Layer(const int channel_num,
              const int map_num=5, const int k=2,
              const float alpha=0.0001, const float beta=0.75);
    void gen_gauss_filter(void);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual int get_batch_size(void);
    virtual int get_channel_num(void);
    virtual vector<int> get_output_map_shape(void);
    virtual vector<int> get_input_map_shape(void);
    vector<int> get_map_shape(void);
    vector<int> get_stlide_shape(void);
    vector<int> get_padding_shape(void);
    vector< vector<MatrixXf> > get_preActivate(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);

private:
    bool trainable = false;
    const string type = "local_response_normalization_layer";
    const bool is_tensor = true;
    int unit_num;
    int batch_size;
    int channel_num;
    int map_height;
    int map_width;
    MatrixXf filter;
    int map_num;
    int t;
    float alpha;
    float beta;
};


LRN_Layer::LRN_Layer(const int channel_num,
                     const int map_num, const int k,
                     const float alpha, const float beta) {
    this->channel_num = channel_num;
    this->map_num = map_num;
    this->t = k;
    this->alpha = alpha;
    this->beta = beta;
}


void LRN_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        for ( int k = 0; k < this->channel_num; ++k ) {
            for ( int h = 0; h < this->map_height; ++h ) {
                for ( int w = 0; w < this->map_width; ++w ) {
                    float tmp = 0.f;
                    int C_min = max(0, k - (int)floor((float)this->map_num / 2.f));
                    int C_max = min(this->channel_num - 1, k + (int)floor(float(this->map_num-1)/2.f));
                    for ( int c = C_min; c <= C_max; ++c ) {
                        tmp += pow(X[n][c](h, w), 2.f);
                    }
                    tmp *= this->alpha;
                    tmp += this->t;
                    this->_activated[n][k](h, w) = X[n][k](h, w) / tmp;
                }
            }
        }
    }
}


void LRN_Layer::calc_delta(const shared_ptr<Layer> &next_layer,
                           const shared_ptr<Layer> &prev_layer) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        for ( int k = 0; k < this->channel_num; ++k ) {
            for ( int h = 0; h < this->map_height; ++h ) {
                for ( int w = 0; w < this->map_width; ++w ) {
                    this->delta[n][k](h, w) = 0.f;
                    int S_min = max(0, k - (int)floor((float)this->map_num/2.f));
                    int S_max = min(this->channel_num-1, k + (int)floor(float(this->map_num-1)/2.f));
                    for ( int s = S_min; s <= S_max; ++s ) {
                        int C_min = max(0, s - (int)floor((float)this->map_num/2.f));
                        int C_max = min(this->channel_num-1, s + (int)floor(float(this->map_num-1)/2.f));
                        float tm = 0.f;
                        for ( int c = C_min; c <= C_max; ++c ) {
                            tm += pow(prev_layer->_activated[n][c](h, w), 2.f);
                        }
                        tm = pow(this->t + tm * this->alpha, float(this->beta+1));
                        this->delta[n][k](h, w) -= next_layer->delta[n][s](h ,w) * prev_layer->_activated[n][s](h, w) / tm;
                    }
                    this->delta[n][k](h, w) *= 2.f * this->alpha * this->beta * prev_layer->_activated[n][k](h, w);
                    int R_min = max(0, k - (int)floor((float)this->map_num / 2.f));
                    int R_max = min(this->channel_num-1, k + (int)floor(float(this->map_num-1)/2.f));
                    float tmp = 0.f;
                    for ( int r = R_min; r <= R_max; ++r ) {
                        tmp += pow(prev_layer->_activated[n][r](h, w), 2.f);
                    }
                    tmp *= this->alpha;
                    tmp += this->t;
                    this->delta[n][k](h, w) += next_layer->delta[n][k](h, w) / pow(tmp, (float)this->beta);
                }
            }
        }
    }
}


void LRN_Layer::allocate_memory(const int batch_size,
                                const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->map_height = prev_layer->get_output_map_shape()[0];
    this->map_width = prev_layer->get_output_map_shape()[1];
    this->unit_num = this->channel_num * this->map_height * this->map_width;

    // activated
    for ( int n = 0; n < this->batch_size; n++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int c = 0; c < this->channel_num; c++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->delta.push_back(tmp_delta);
    }
}


bool LRN_Layer::get_trainable(void) { return this->trainable; }
string LRN_Layer::get_type(void) { return this->type; }
bool LRN_Layer::get_is_tensor(void) { return this->is_tensor; }
int LRN_Layer::get_unit_num(void) { return this->unit_num; }
int LRN_Layer::get_batch_size(void) { return this->batch_size; }
int LRN_Layer::get_channel_num(void) { return this->channel_num; }
vector<int> LRN_Layer::get_input_map_shape(void) {
    vector<int> map_shape{ this->map_height, this->map_width };
    return map_shape;
}
vector<int> LRN_Layer::get_output_map_shape(void) {
    vector<int> map_shape{ this->map_height, this->map_width };
    return map_shape;
}
vector< vector<MatrixXf> > LRN_Layer::get_activated(void) {
    return this->_activated;
}
vector<vector <MatrixXf> > LRN_Layer::get_delta(void) {
    return this->delta;
}


#endif // INCLUDE_local_response_normalization_layer_h_
