#ifndef INCLUDE_max_pooling_layer_h_
#define INCLUDE_max_pooling_layer_h_

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
using Eigen::MatrixXf;
using std::shared_ptr;


class Max_Pooling_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const vector<vector<MatrixXf> > next_delta,
                            const vector<vector<MatrixXf> > prev_activated);
    virtual void allocate_memory(const int batch_size, const int prev_height, const int prev_width);

    Max_Pooling_Layer(const int channel_num,
                      const int filter_height, const int filter_width,
                      const int stlide_height=1, const int stlide_width=1,
                      const int padding_height=0, const int padding_width=0);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual int get_batch_size(void);
    virtual int get_channel_num(void);
    virtual vector<int> get_output_map_shape(void);
    virtual vector<int> get_input_map_shape(void);
    vector<int> get_map_shape(void);
    vector<int> get_filter_shape(void);
    vector<int> get_stlide_shape(void);
    vector<int> get_padding_shape(void);
    vector< vector<MatrixXf> > get_preActivate(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);

private:
    bool trainable = false;
    string type = "max_pooling_layer";
    int batch_size;
    int channel_num;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int filter_height;
    int filter_width;
    int stlide_height;
    int stlide_width;
    int padding_height;
    int padding_width;
    vector< vector<MatrixXf> > delta;
};


Max_Pooling_Layer::Max_Pooling_Layer(const int channel_num,
                                     const int filter_height, const int filter_width,
                                     const int stlide_height, const int stlide_width,
                                     const int padding_height, const int padding_width) {
    this->channel_num = channel_num;
    this->filter_height = filter_height;
    this->filter_width = filter_width;
    this->stlide_height = stlide_height;
    this->stlide_width = stlide_width;
    this->padding_height = padding_height;
    this->padding_width = padding_width;
}


void Max_Pooling_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        int h = 0;
        int w = 0;
        for ( int k = 0; k < this->channel_num; k++ ) {
            for ( int p = 0; p < this->output_height; p++ ) {
                h = p * this->stlide_height - this->padding_height;
                for ( int q = 0; q < this->output_width; q++ ){
                    w = q * this->stlide_width - this->padding_width;
                    this->_activated[n][k](p, q) = X[n][k].block(h, w, this->filter_height, this->filter_width).maxCoeff();
                }
            }
        }
    }
}


void Max_Pooling_Layer::calc_delta(const vector<vector<MatrixXf> > next_delta,
                                   const vector<vector<MatrixXf> > prev_activated) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        int P_max = 0;
        int P_min = 0;
        int Q_max = 0;
        int Q_min = 0;
        int r = 0;
        int s = 0;
        for ( int c = 0; c < this->channel_num; c++ ) {
            for ( int h = 0; h < this->input_height; h++ ) {
                for ( int w = 0; w < this->input_width; w++ ) {
                    this->delta[n][c](h, w) = 0.f;
                    P_min = std::max(0,
                        (int)ceil((h - this->filter_height + 1 + this->padding_height) / this->stlide_height));
                    P_max = std::min(this->output_height-1,
                        (int)floor((h + this->padding_height) / this->stlide_height));
                    for ( int p = P_min; p <= P_max; p++ ) {
                        r = h - p * this->stlide_height + this->padding_height;
                        Q_min = std::max(0,
                            (int)ceil((w - this->filter_width + 1 + this->padding_width) / this->stlide_width));
                        Q_max = std::min(this->output_width-1,
                            (int)floor((w + this->padding_width) / this->stlide_width));
                        for ( int q = Q_min; q <= Q_max; q++ ) {
                            s = w - q * this->stlide_width + this->padding_width;
                            if ( this->_activated[n][c](p, q) == prev_activated[n][c](h, w) ) {
                                this->delta[n][c](h, w) += next_delta[n][c](p, q);
                            }
                        }
                    }
                }
            }
        }
    }
}


void Max_Pooling_Layer::allocate_memory(const int batch_size,
                                        const int input_height,
                                        const int input_width) {
    this->batch_size = batch_size;
    this->input_height = input_height;
    this->input_width = input_width;
    this->output_height = ceil((input_height - this->filter_height + 1 + 2 * this->padding_height) / this->stlide_height);
    this->output_width = ceil((input_width - this->filter_width + 1 + 2 * this->padding_width) / this->stlide_width);


    // activated & delta
    for ( int n = 0; n < this->batch_size; n++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int c = 0; c < this->channel_num; c++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->output_height, this->output_width));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->input_height, this->input_width));
        }
        this->delta.push_back(tmp_delta);
    }
}


bool Max_Pooling_Layer::get_trainable(void) { return this->trainable; }
string Max_Pooling_Layer::get_type(void) { return this->type; }
int Max_Pooling_Layer::get_batch_size(void) { return this->batch_size; }
int Max_Pooling_Layer::get_channel_num(void) { return this->channel_num; }
vector<int> Max_Pooling_Layer::get_input_map_shape(void) {
    vector<int> map_shape{ this->input_height, this->input_width };
    return map_shape;
}
vector<int> Max_Pooling_Layer::get_output_map_shape(void) {
    vector<int> map_shape{ this->output_height, this->output_width };
    return map_shape;
}
vector<int> Max_Pooling_Layer::get_filter_shape(void) {
    vector<int> filter_shape{ this->filter_height, this->filter_width };
    return filter_shape;
}
vector<int> Max_Pooling_Layer::get_stlide_shape(void) {
    vector<int> stlide_shape{ this->stlide_height, this->stlide_width };
    return stlide_shape;
}
vector<int> Max_Pooling_Layer::get_padding_shape(void) {
    vector<int> padding_shape{ this->padding_height, this->padding_width };
    return padding_shape;
}
vector< vector<MatrixXf> > Max_Pooling_Layer::get_activated(void) {
    return this->_activated;
}
vector<vector <MatrixXf> > Max_Pooling_Layer::get_delta(void) {
    return this->delta;
}


#endif // INCLUDE_max_pooling_layer_h_
