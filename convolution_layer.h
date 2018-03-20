#ifndef INCLUDE_convolution_layer_h_
#define INCLUDE_convolution_layer_h_

#include <iostream>
#include <functional>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "my_math.h"
#include "layer.h"
#include <chrono>

using std::function;
using std::cout;
using std::endl;
using std::make_shared;
using std::max;
using std::min;
using std::shared_ptr;
using Eigen::MatrixXf;


class Convolution_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_differential(const shared_ptr<Layer> &prev_layer,
                                   const shared_ptr<Layer> &next_layer);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer);
    virtual void allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer);

    Convolution_Layer(const int prev_ch, const int ch,
                      const int filter_height, const int filter_width,
                      const int stlide_height=1, const int stlide_width=1,
                      const int padding_height=0, const int padding_width=0,
                      const string initializer="Xavier");

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual int get_batch_size(void);
    virtual int get_channel_num(void);
    virtual int get_prev_channel_num(void);
    vector<int>  get_filter_shape(void);
    vector<int> get_stlide_shape(void);
    vector<int> get_padding_shape(void);
    vector<int> get_input_map_shape(void);
    vector<int> get_output_map_shape(void);
    vector<vector <MatrixXf> > get_W(void);
    MatrixXf get_b(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);
    virtual vector< vector<MatrixXf> > get_dE_dW(void);
    MatrixXf get_dE_db(void);

    // setter
    virtual void set_batch_size(const int batch_size, const shared_ptr<Layer> &prev_layer);
    // virtual void set_W(MatrixXf);
    // virtual void set_b(MatrixXf);
    virtual void set_delta(const vector< vector<MatrixXf> > delta);


private:
    bool trainable = true;
    const string type = "convolution_layer";
    const bool is_tensor = true;
    int unit_num;
    // Parameters tracked during learning
    vector <vector<MatrixXf> > _d_f;
    // vector<vector <MatrixXf> > W;   // NOTE:convolutionしか使ってない問題
    // vector< vector<MatrixXf> > dE_dW;   // NOTE:convolutionしか使ってない問題
    // MatrixXf dE_db; // NOTE:convolutionしか使ってない問題
    // MatrixXf b; // NOTE:convolutionしか使ってない問題
    // Parameters specified at first
    int batch_size;
    int channel_num;
    int prev_channel_num;
    int filter_height;
    int filter_width;
    int stlide_height;
    int stlide_width;
    int padding_height;
    int padding_width;
    int W_shape[4];
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    string initializer;
    function<MatrixXf(MatrixXf)> f;
    function<MatrixXf(MatrixXf)> d_f;
    // Storage during learning
    MatrixXf _preActivate;
};


// NOTE: Wの初期化どうしよう問題
// HeやXavierのせいで一旦allocate_memoryに行ってしまったが、これだと一様分布で初期化できなくなってしまう。
Convolution_Layer::Convolution_Layer(const int prev_ch, const int ch,
                                     const int filter_height, const int filter_width,
                                     const int stlide_height, const int stlide_width,
                                     const int padding_height, const int padding_width,
                                     const string initializer) {
    /*
        W[channel, prev_channel, row, col]
        use bias common to each unit for each filter
    */
    this->W_shape[0] = ch; this->W_shape[1] = prev_ch; this->W_shape[2] = filter_height; this->W_shape[3] = filter_width;
    this->channel_num = ch;
    this->prev_channel_num = prev_ch;
    this->filter_height = filter_height;
    this->filter_width = filter_width;
    this->stlide_height = stlide_height;
    this->stlide_width = stlide_width;
    this->padding_height = padding_height;
    this->padding_width = padding_width;
    this->initializer = initializer;
}


void Convolution_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        int h = 0;
        int w = 0;
        int part_h = 0;
        int part_w = 0;
        shared_ptr<MatrixXf> X_ptr;
        shared_ptr<MatrixXf> W_ptr;
        for ( int k = 0; k < this->channel_num; ++k ) {
            // convolute
            for ( int p = 0; p < this->output_height; ++p ) {
                part_h = p * this->stlide_height - this->padding_height;
                for ( int q = 0; q < this->output_width; ++q ){
                    part_w = q * this->stlide_width - this->padding_width;
                    this->_activated[n][k](p, q) = this->b(0, k);
                    for ( int c = 0; c < this->prev_channel_num; ++c ) {
                        X_ptr = make_shared<MatrixXf>(X[n][c]);
                        W_ptr = make_shared<MatrixXf>(this->W[k][c]);
                        for ( int r = 0; r < this->filter_height; ++r ) {
                            h = part_h + r;
                            for ( int s = 0; s < this->filter_width; ++s ) {
                                w = part_w + s;
                                this->_activated[n][k](p, q) += (*X_ptr)(h, w) * (*W_ptr)(r, s);
                            }
                        }
                    }
                }
            }
        }
    }
}


void Convolution_Layer::calc_delta(const shared_ptr<Layer> &next_layer) {
    shared_ptr< vector< vector<MatrixXf> > > next_delta
        = make_shared< vector< vector<MatrixXf> > >(next_layer->delta);
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        int P_max = 0;
        int P_min = 0;
        int Q_max = 0;
        int Q_min = 0;
        int r = 0;
        int s = 0;
        int part_r = 0;
        int part_s = 0;
        shared_ptr<MatrixXf> next_delta_ptr;
        shared_ptr<MatrixXf> W_ptr;
        for ( int c = 0; c < this->prev_channel_num; ++c ) {
            for ( int h = 0; h < this->input_height; ++h ) {
                P_min = max(0,
                    (int)ceil((float)(h - this->filter_height + 1 + this->padding_height) / (float)this->stlide_height));
                P_max = min(this->output_height-1,
                    (int)floor((float)(h + this->padding_height) / (float)this->stlide_height));
                part_r = h + this->padding_height;
                for ( int w = 0; w < this->input_width; ++w ) {
                    this->delta[n][c](h, w) = 0.f;
                    Q_min = max(0,
                        (int)ceil((float)(w - this->filter_width + 1 + this->padding_width) / (float)this->stlide_width));
                    Q_max = min(this->output_width-1,
                        (int)floor((float)(w + this->padding_width) / (float)this->stlide_width));
                    part_s = w + this->padding_width;
                    for ( int k = 0; k < this->channel_num; ++k ) {
                        next_delta_ptr = make_shared<MatrixXf>((*next_delta)[n][k]);
                        W_ptr = make_shared<MatrixXf>(this->W[k][c]);
                        for ( int p = P_min; p <= P_max; ++p ) {
                            r = part_r - p * this->stlide_height;
                            for ( int q = Q_min; q <= Q_max; ++q ) {
                                s = part_s - q * this->stlide_width;
                                this->delta[n][c](h, w)
                                    += (*next_delta_ptr)(p, q) * (*W_ptr)(r, s);
                            }
                        }
                    }
                }
            }
        }
    }
}


void Convolution_Layer::calc_differential(const shared_ptr<Layer> &prev_layer,
                                          const shared_ptr<Layer> &next_layer) {

    shared_ptr< vector< vector<MatrixXf> > > next_delta
        = make_shared< vector< vector<MatrixXf> > >(next_layer->delta);
    shared_ptr< vector< vector<MatrixXf> > > prev_activated
        = make_shared< vector< vector<MatrixXf> > >(prev_layer->_activated);
    // W
    this->dE_db = MatrixXf::Zero(1, this->channel_num);
    #pragma omp parallel for
    for ( int k = 0; k < this->channel_num; ++k ) {
        int h = 0;
        int w = 0;
        shared_ptr<MatrixXf> next_delta_ptr;
        shared_ptr<MatrixXf> prev_activated_ptr;
        for ( int c = 0; c < this->prev_channel_num; ++c ) {
            this->dE_dW[k][c] = MatrixXf::Zero(this->filter_height, this->filter_width);
            for ( int r = 0; r < this->filter_height; ++r ) {
                for ( int s = 0; s < this->filter_width; ++s ) {
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        prev_activated_ptr = make_shared<MatrixXf>((*prev_activated)[n][c]);
                        next_delta_ptr = make_shared<MatrixXf>((*next_delta)[n][k]);
                        for ( int q = 0; q < this->output_height; ++q ) {
                            w = q * this->stlide_width + s - this->padding_width;
                            for ( int p = 0; p < this->output_width; ++p ) {
                                h = p * this->stlide_height + r - this->padding_height;
                                this->dE_dW[k][c](r, s) += (*next_delta_ptr)(p, q) * (*prev_activated_ptr)(h, w);
                                this->dE_db(0, k) += (*next_delta_ptr)(p, q);
                            }
                        }
                    }
                }
            }
            this->dE_dW[k][c] /= (float)this->batch_size;
        }
    }
    this->dE_db /= (float)this->batch_size;
}


void Convolution_Layer::allocate_memory(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->input_height = prev_layer->get_output_map_shape()[0];
    this->input_width = prev_layer->get_output_map_shape()[0];
    this->output_height = ceil((this->input_height - this->filter_height + 1 + 2 * this->padding_height) / this->stlide_height);
    this->output_width = ceil((this->input_width - this->filter_width + 1 + 2 * this->padding_width) / this->stlide_width);
    this->unit_num = this->channel_num * this->output_height * this->output_width;

    // Define weight and bias at random.
    if ( initializer == "Xavier" ) {
        this->W = gauss_rand(W_shape, 0.f, sqrt(1.f/(this->input_height*this->input_width*this->prev_channel_num)));
        this->b = gauss_rand(this->channel_num, 0.f, sqrt(1.f/(this->input_height*this->input_width*this->prev_channel_num)));
    } else if ( initializer == "He" ) {
        this->W = gauss_rand(W_shape, 0.f, sqrt(2.f/(this->input_height*this->input_width*this->prev_channel_num)));
        this->b = gauss_rand(this->channel_num, 0.f, sqrt(2.f/(this->input_height*this->input_width*this->prev_channel_num)));
    } else if ( initializer == "uniform" ) {
        this->W = uniform_rand(this->W_shape, -0.01, 0.01);
        this->b = uniform_rand(this->channel_num, -0.01, 0.01);
    }

    // preActivate & activated
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->output_height, this->output_width));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->prev_channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->input_height, this->input_width));
        }
        this->delta.push_back(tmp_delta);
    }

    // dE_db
    this->dE_db.resize(1, this->channel_num);

    // dE_dW
    for ( int i = 0; i < this->channel_num; i++ ) {
        vector<MatrixXf> tmp_dE_dW;
        for ( int j = 0; j < this->prev_channel_num; j++ ) {
            tmp_dE_dW.push_back(MatrixXf::Zero(this->filter_height, this->filter_width));
        }
        this->dE_dW.push_back(tmp_dE_dW);
    }

    // _d_f (NOTE::活性化レイヤーを個別に作ったほうがいいかも)
    for ( int n = 0; n < this->batch_size; n++ ) {
        vector<MatrixXf> tmp_d_f;
        for ( int c = 0; c < this->channel_num; c++ ) {
            tmp_d_f.push_back(MatrixXf::Zero(this->output_height, this->output_width));
        }
        this->_d_f.push_back(tmp_d_f);
    }
}


bool Convolution_Layer::get_trainable(void) { return this->trainable; }
string Convolution_Layer::get_type(void) { return this->type; }
bool Convolution_Layer::get_is_tensor(void) { return this->is_tensor; }
int Convolution_Layer::get_unit_num(void) { return this->unit_num; }
int Convolution_Layer::get_batch_size(void) { return this->batch_size; }
int Convolution_Layer::get_channel_num(void) { return this->channel_num; }
int Convolution_Layer::get_prev_channel_num(void) { return this->prev_channel_num; }
vector<int> Convolution_Layer::get_filter_shape(void) {
    vector<int> filter_shape; filter_shape.resize(4);
    filter_shape[0] = this->W_shape[0]; filter_shape[1] = this->W_shape[1];
    filter_shape[2] = this->W_shape[2]; filter_shape[3] = this->W_shape[3];
    return filter_shape; }
vector<int> Convolution_Layer::get_stlide_shape(void) {
    vector<int> stlide_shape{ this->stlide_height, this->stlide_width };
    return stlide_shape;
}
vector<int> Convolution_Layer::get_padding_shape(void) {
    vector<int> padding_shape{ this->padding_height, this->padding_width };
    return padding_shape;
}
vector<int> Convolution_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape{ this->input_height, this->input_width };
    return input_map_shape;
}
vector<int> Convolution_Layer::get_output_map_shape(void) {
    vector<int> output_map_shape{ this->output_height, this->output_width };
    return output_map_shape;
}
vector<vector <MatrixXf> > Convolution_Layer::get_W(void) { return this->W; }
MatrixXf Convolution_Layer::get_b(void) {return this->b; }
vector< vector<MatrixXf> > Convolution_Layer::get_activated(void) { return this->_activated; }
vector< vector<MatrixXf> > Convolution_Layer::get_delta(void) { return this->delta; }
vector< vector<MatrixXf> > Convolution_Layer::get_dE_dW(void) { return this->dE_dW; }
MatrixXf Convolution_Layer::get_dE_db(void) {return this->dE_db; }


void Convolution_Layer::set_batch_size(const int batch_size, const shared_ptr<Layer> &prev_layer) {
    this->allocate_memory(batch_size, prev_layer);
}

void Convolution_Layer::set_delta(const vector< vector<MatrixXf> > delta) {
    this->delta = delta;
}


#endif // INCLUDE_convolution_layer_h_
