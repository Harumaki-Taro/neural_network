#ifndef INCLUDE_local_constrast_normalization_layer_h_
#define INCLUDE_local_constrast_normalization_layer_h_

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


class LCN_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    void substractive_norm_FP(const vector< vector<MatrixXf> > X);
    void divisive_norm_FP(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer,
                            const shared_ptr<Layer> &prev_layer);
    void substractive_norm_BP(const vector< vector<MatrixXf> > next_delta);
    void divisive_norm_BP(const vector< vector<MatrixXf> > X,
                          const vector< vector<MatrixXf> > next_delta);
    virtual void allocate_memory(const int batch_size,
                                 const shared_ptr<Layer> &prev_layer);

    LCN_Layer(const int channel_num,
                 const int filter_height, const int filter_width,
                 const string norm_type="divisive",
                 const string filter_type="gauss");
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
    vector<int> get_filter_shape(void);
    vector<int> get_stlide_shape(void);
    vector<int> get_padding_shape(void);
    vector< vector<MatrixXf> > get_preActivate(void);
    virtual vector< vector<MatrixXf> > get_activated(void);
    virtual vector<vector <MatrixXf> > get_delta(void);

private:
    bool trainable = false;
    const string type = "local_constract_normalization_layer";
    const bool is_tensor = true;
    int unit_num;
    int batch_size;
    int channel_num;
    int map_height;
    int map_width;
    int filter_height;
    int filter_width;
    float eps = 0.000001;
    float reciprocal_channel_num;
    string filter_type="gauss";
    string norm_type="divisive";
    MatrixXf filter;
    vector<MatrixXf> mu;
    vector<MatrixXf> sig2;
    vector<MatrixXf> __sig2;
    vector< vector<MatrixXf> > d_mu;
    vector< vector< vector<MatrixXf> > > d_sig2;
    vector< vector<MatrixXf> > act_SN;
    vector< vector<MatrixXf> > act_DN;
    vector< vector<MatrixXf> > delta_SN;
    vector< vector<MatrixXf> > delta_DN;
};


LCN_Layer::LCN_Layer(const int channel_num,
                     const int filter_height, const int filter_width,
                     const string norm_type,
                     const string filter_type) {
    this->channel_num = channel_num;
    this->reciprocal_channel_num = 1.f / (float)this->channel_num;
    this->filter_height = filter_height;
    this->filter_width = filter_width;
    if ( this->filter_width != this->filter_height ) {
        cout << "LCN_Layerのカーネルの縦横のサイズが一致していません" << endl;
    }
    if ( this->filter_width % 2 == 0 ) {
        cout << "LCN_Layerのカーネルのサイズが偶数になっています" << endl;
    }

    if ( norm_type == "divisive" || norm_type == "substractive" ) {
        this->norm_type = norm_type;
    } else {
        cout << "指定の正規化は実装されていません" << endl;
        exit(1);
    }

    if ( filter_type == "gauss" ) {
        this->filter_type = filter_type;
        this->gen_gauss_filter();
    } else if ( filter_type == "moving_average" ) {
        cout << "指定のフィルタタイプは実装されていません" << endl;
        exit(1);
    } else if ( filter_type == "median" ) {
        cout << "指定のフィルタタイプは実装されていません" << endl;
        exit(1);
    } else if ( filter_type == "bilateral" ) {
        cout << "指定のフィルタタイプは実装されていません" << endl;
        exit(1);
    } else {
        cout << "指定のフィルタタイプは実装されていません" << endl;
        exit(1);
    }
}


void LCN_Layer::gen_gauss_filter(void) {
    this->filter = MatrixXf::Zero(this->filter_height, this->filter_width);
    if ( this->filter_height == 1 ) {
        cout << "LCN_Layerのカーネルのサイズが無効です" << endl;
    } else if ( this->filter_height == 3 ) {
        this->filter << 1.f, 2.f, 1.f,
                        2.f, 4.f, 2.f,
                        1.f, 2.f, 1.f;
        this->filter /= 16.f;
    } else if ( this->filter_height == 5 ) {
        this->filter <<  1.f,  4.f,  6.f,  4.f,  1.f,
                         4.f, 16.f, 24.f, 16.f,  4.f,
                         6.f, 24.f, 36.f, 24.f,  6.f,
                         4.f, 16.f, 24.f, 16.f,  4.f,
                         1.f,  4.f,  6.f,  4.f,  1.f;
        this->filter /= 256.f;
    } else if ( this->filter_height == 7 ) {
        this->filter <<   1.f,   6.f,  15.f,  20.f,  15.f,   6.f,   1.f,
                          6.f,  36.f,  90.f, 120.f,  90.f,  36.f,   6.f,
                         15.f,  90.f, 225.f, 300.f, 225.f,  90.f,  16.f,
                         20.f, 120.f, 300.f, 400.f, 300.f, 120.f,  20.f,
                         15.f,  90.f, 225.f, 300.f, 225.f,  90.f,  16.f,
                          6.f,  36.f,  90.f, 120.f,  90.f,  36.f,   6.f,
                          1.f,   6.f,  15.f,  20.f,  15.f,   6.f,   1.f;
        this->filter /= 4096.f;
    } else {
        cout << "LCN_Layerのカーネルのサイズが無効です" << endl;
    }
}


void LCN_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    if ( this->norm_type == "substractive") {
        this->substractive_norm_FP(X);
    } else {
        this->divisive_norm_FP(X);
    }
}


void LCN_Layer::substractive_norm_FP(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int h = 0; h < this->map_height; h++ ) {
            int _P_min = h - (int)(floor((float)this->filter_height) / 2.f);
            int P_min = max(0, _P_min);
            int P_max = min(this->map_height - 1,
                            h - 1 + (int)(floor(((float)this->filter_height + 1.f) / 2.f)));
            for ( int w = 0; w < this->map_width; w++ ) {
                this->mu[n](h, w) = 0.f;
                int _Q_min = w - (int)(floor((float)this->filter_width) / 2.f);
                int Q_min = max(0, _Q_min);
                int Q_max = min(this->map_width - 1,
                                w - 1 + (int)(floor(((float)this->filter_width + 1.f) / 2.f)));
                for ( int c = 0; c < this->channel_num; c++ ) {
                    for ( int p = P_min; p <= P_max; p++ ) {
                        for ( int q = Q_min; q <= Q_max; q++ ) {
                            this->mu[n](h, w) += this->filter(p-_P_min, q-_Q_min) * X[n][c](p, q);
                        }
                    }
                }
            }
        }
        this->mu[n] *= reciprocal_channel_num;

        for ( int c = 0; c < this->channel_num; c++ ) {
            this->_activated[n][c] = X[n][c] - this->mu[n];
        }
    }
}


void LCN_Layer::divisive_norm_FP(const vector< vector<MatrixXf> > X) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; n++ ) {
        for ( int h = 0; h < this->map_height; h++ ) {
            int _P_min = h - (int)(floor((float)this->filter_height) / 2.f);
            int P_min = max(0, _P_min);
            int P_max = min(this->map_height - 1,
                            h - 1 + (int)(floor(((float)this->filter_height + 1.f) / 2.f)));
            for ( int w = 0; w < this->map_width; w++ ) {
                this->mu[n](h, w) = 0.f;
                int _Q_min = w - (int)(floor((float)this->filter_width) / 2.f);
                int Q_min = max(0, _Q_min);
                int Q_max = min(this->map_width - 1,
                                w - 1 + (int)(floor(((float)this->filter_width + 1.f) / 2.f)));
                for ( int c = 0; c < this->channel_num; c++ ) {
                    for ( int p = P_min; p <= P_max; p++ ) {
                        for ( int q = Q_min; q <= Q_max; q++ ) {
                            this->mu[n](h, w) += this->filter(p-_P_min, q-_Q_min) * X[n][c](p, q);
                        }
                    }
                }
            }
        }
        this->mu[n] *= this->reciprocal_channel_num;

        for ( int h = 0; h < this->map_height; h++ ) {
            int _P_min = h - (int)(floor((float)this->filter_height) / 2.f);
            int P_min = max(0, _P_min);
            int P_max = min(this->map_height - 1,
                            h - 1 + (int)(floor(((float)this->filter_height + 1.f) / 2.f)));
            for ( int w = 0; w < this->map_width; w++ ) {
                this->sig2[n](h, w) = 0.f;
                int _Q_min = w - (int)(floor((float)this->filter_width) / 2.f);
                int Q_min = max(0, _Q_min);
                int Q_max = min(this->map_width - 1,
                                w - 1 + (int)(floor(((float)this->filter_width + 1.f) / 2.f)));

                for ( int c = 0; c < this->channel_num; c++ ) {
                    for ( int p = P_min; p <= P_max; p++ ) {
                        for ( int q = Q_min; q <= Q_max; q++ ) {
                            this->sig2[n](h, w) += this->filter(p-_P_min, q-_Q_min)
                                                * pow(X[n][c](p, q) - this->mu[n](h, w), 2.f);
                        }
                    }
                }
                this->sig2[n](h, w) *= this->reciprocal_channel_num;
            }
        }

        for ( int h = 0; h < this->map_height; ++h ) {
            for ( int w = 0; w < this->map_width; ++w ) {
                this->__sig2[n](h, w) = 1.f / sqrt(this->eps + this->sig2[n](h, w));
            }
        }

        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int h = 0; h < this->map_height; ++h ) {
                for ( int w = 0; w < this->map_width; ++w ) {
                    this->_activated[n][c](h, w) = (X[n][c](h, w) - this->mu[n](h, w)) * this->__sig2[n](h, w);
                }
            }
        }
    }
}


void LCN_Layer::calc_delta(const shared_ptr<Layer> &next_layer,
                           const shared_ptr<Layer> &prev_layer) {
    this->substractive_norm_BP(next_layer->delta);

    if ( this->norm_type == "substractive") {
        this->substractive_norm_BP(next_layer->delta);
    } else {
        this->divisive_norm_BP(prev_layer->_activated,
                               next_layer->delta);
    }
}


void LCN_Layer::substractive_norm_BP(const vector< vector<MatrixXf> > next_delta) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int h = 0; h < this->map_height; ++h ) {
                int A_min = max(0, h + 1 - (int)floor((float(this->filter_height + 1))/2.f));
                int A_max = min(this->map_height - 1, h + (int)floor((float(this->filter_height))/2.f));
                for ( int w = 0; w < this->map_width; ++w ) {
                    int B_min = max(0, w + 1 - (int)floor((float(this->filter_width + 1))/2.f));
                    int B_max = min(this->map_width - 1, w + (int)floor((float(this->filter_width))/2.f));
                    this->delta[n][c](h, w) = next_delta[n][c](h, w);
                    for ( int k = 0; k < this->channel_num; ++k ) {
                        for ( int a = A_min; a <= A_max; ++a ) {
                            for ( int b = B_min; b <= B_max; ++b ) {
                                this->delta[n][c](h, w) -= next_delta[n][k](a, b) * this->d_mu[h][w](a, b);
                            }
                        }
                    }
                }
            }
        }
    }
}


void LCN_Layer::divisive_norm_BP(const vector< vector<MatrixXf> > X,
                                    const vector< vector<MatrixXf> > next_delta) {
    #pragma omp parallel for
    for ( int n = 0; n < this->batch_size; ++n ) {
        // d_sig2
        for ( int c = 0; c < this->channel_num; c++ ) {
            for ( int h = 0; h < this->map_height; h++ ) {
                int A_min = max(0, h + 1 - (int)floor((float(this->filter_height + 1))/2.f));
                int A_max = min(this->map_height - 1, h + (int)floor((float(this->filter_height))/2.f));
                for ( int w = 0; w < this->map_width; w++ ) {
                    int B_min = max(0, w + 1 - (int)floor((float(this->filter_width + 1))/2.f));
                    int B_max = min(this->map_width - 1, w + (int)floor((float(this->filter_width))/2.f));
                    for ( int a = A_min; a <= A_max; ++a ) {
                        int r = (int)floor(float(this->filter_height)/2.f) + h - a;
                        int P_min = max(0, a - (int)floor(float(this->filter_height)/2.f));
                        int P_max = min(this->map_height - 1,
                                        a - 1 + (int)floor(float(this->filter_height + 1)/2.f));
                        for ( int b = B_min; b <= B_max; ++b ) {
                            int s = (int)floor(float(this->filter_width) / 2.f) + w - b;
                            int Q_min = max(0, b - (int)floor(float(this->filter_width)/2.f));
                            int Q_max = min(this->map_width - 1,
                                            b - 1 + (int)floor(float(this->filter_width + 1)/2.f));
                            this->d_sig2[c][h][w](a, b) = 0.f;
                            for ( int k = 0; k < this->channel_num; ++k ) {
                                for ( int p = P_min; p <= P_max; ++p ) {
                                    int u = p - a + (int)floor(float(this->filter_height) / 2.f);
                                    for ( int q = Q_min; q <= Q_max; ++q ) {
                                        int v = q - b + (int)floor(float(this->filter_width) / 2.f);
                                        this->d_sig2[c][h][w](a, b) -= this->filter(u, v)
                                                                    * (X[n][k](p, q) - this->mu[n](a, b))
                                                                    * this->d_mu[h][w](a, b);
                                    }
                                }
                            }
                            this->d_sig2[c][h][w](a, b) += this->filter(r, s) * (X[n][c](h, w) - this->mu[n](a, b));
                            this->d_sig2[c][h][w](a, b) *= 2.f * this->reciprocal_channel_num;
                        }
                    }
                }
            }
        }

        // delta
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int h = 0; h < this->map_height; ++h ) {
                int A_min = max(0, h + 1 - (int)floor((float(this->filter_height + 1))/2.f));
                int A_max = min(this->map_height - 1, h + (int)floor((float(this->filter_height))/2.f));
                for ( int w = 0; w < this->map_width; ++w ) {
                    int B_min = max(0, w + 1 - (int)floor((float(this->filter_width + 1))/2.f));
                    int B_max = min(this->map_width - 1, w + (int)floor((float(this->filter_width))/2.f));
                    this->delta[n][c](h, w) = 0.f;
                    for ( int k = 0; k < this->channel_num; ++k ) {
                        for ( int a = A_min; a <= A_max; ++a ) {
                            for ( int b = B_min; b <= B_max; ++b ) {
                                float _dx_dx = this->d_mu[h][w](a, b) / (this->eps + this->sig2[n](a, b))
                                             + 0.5f * (X[n][k](a, b) - this->mu[n](a, b)) * this->d_sig2[c][h][w](a, b)
                                             / pow(this->eps + this->sig2[n](a, b), 2.f);
                                _dx_dx *= sqrt(this->eps + this->sig2[n](a, b));
                                this->delta[n][c](h, w) -= next_delta[n][k](a, b) * _dx_dx;
                            }
                        }
                    }
                    this->delta[n][c](h, w) += next_delta[n][c](h, w) * this->__sig2[n](h, w);
                }
            }
        }
    }
}


void LCN_Layer::allocate_memory(const int batch_size,
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

    // act_SN
    for ( int n = 0; n < this->batch_size; n++ ) {
        vector<MatrixXf> tmp_act_SN;
        for ( int c = 0; c < this->channel_num; c++ ) {
            tmp_act_SN.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->act_SN.push_back(tmp_act_SN);
    }

    // act_DN
    for ( int n = 0; n < this->batch_size; n++ ) {
        vector<MatrixXf> tmp_act_DN;
        for ( int c = 0; c < this->channel_num; c++ ) {
            tmp_act_DN.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->act_DN.push_back(tmp_act_DN);
    }

    // delta
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->delta.push_back(tmp_delta);
    }

    // delta_SN
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta_SN;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta_SN.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->delta_SN.push_back(tmp_delta_SN);
    }

    // delta_DN
    for ( int i = 0; i < this->batch_size; i++ ) {
        vector<MatrixXf> tmp_delta_DN;
        for ( int j = 0; j < this->channel_num; j++ ) {
            tmp_delta_DN.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->delta_DN.push_back(tmp_delta_DN);
    }

    // mu
    for ( int n = 0; n < this->batch_size; n++ ) {
        this->mu.push_back(MatrixXf::Zero(this->map_height, this->map_width));
    }

    // d_mu
    for ( int i = 0; i < this->map_height; i++ ) {
        vector<MatrixXf> tmp_d_mu;
        for ( int j = 0; j < this->map_width; j++ ) {
            tmp_d_mu.push_back(MatrixXf::Zero(this->map_height, this->map_width));
        }
        this->d_mu.push_back(tmp_d_mu);
    }

    for ( int h = 0; h < this->map_height; ++h ) {
        int A_min = max(0, h + 1 - (int)floor((float(this->filter_height+1))/2.f));
        int A_max = min(this->map_height - 1, h + (int)floor((float(this->filter_height))/2.f));
        for ( int w = 0; w < this->map_width; ++w ) {
            int B_min = max(0, w + 1 - (int)floor((float(this->filter_width+1))/2.f));
            int B_max = min(this->map_width - 1, w + (int)floor((float(this->filter_width))/2.f));
            for ( int a = A_min; a <= A_max; ++a ) {
                int r = (int)floor(float(this->filter_height)/2.f) + h - a;
                for ( int b = B_min; b <= B_max; ++b ) {
                    int s = (int)floor(float(this->filter_width)/2.f) + w - b;
                    this->d_mu[h][w](a, b) = this->filter(r, s);
                }
            }
            this->d_mu[h][w] *= this->reciprocal_channel_num;
        }
    }

    // sig2
    for ( int n = 0; n < this->batch_size; n++ ) {
        this->sig2.push_back(MatrixXf::Zero(this->map_height, this->map_width));
    }

    // __sig2 (sig2^(-1/2))
    for ( int n = 0; n < this->batch_size; n++ ) {
        this->__sig2.push_back(MatrixXf::Zero(this->map_height, this->map_width));
    }

    // d_sig2
    for ( int i = 0; i < this->channel_num; i++ ) {
        vector< vector<MatrixXf> > tmp_d_sig2;
        for ( int j = 0; j < this->map_height; j++ ) {
            vector<MatrixXf> tmp_tmp_d_sig2;
            for ( int k = 0; k < this->map_width; k++ ) {
                tmp_tmp_d_sig2.push_back(MatrixXf::Zero(this->map_height, this->map_width));
            }
            tmp_d_sig2.push_back(tmp_tmp_d_sig2);
        }
        this->d_sig2.push_back(tmp_d_sig2);
    }
}


bool LCN_Layer::get_trainable(void) { return this->trainable; }
string LCN_Layer::get_type(void) { return this->type; }
bool LCN_Layer::get_is_tensor(void) { return this->is_tensor; }
int LCN_Layer::get_unit_num(void) { return this->unit_num; }
int LCN_Layer::get_batch_size(void) { return this->batch_size; }
int LCN_Layer::get_channel_num(void) { return this->channel_num; }
vector<int> LCN_Layer::get_input_map_shape(void) {
    vector<int> map_shape{ this->map_height, this->map_width };
    return map_shape;
}
vector<int> LCN_Layer::get_output_map_shape(void) {
    vector<int> map_shape{ this->map_height, this->map_width };
    return map_shape;
}
vector<int> LCN_Layer::get_filter_shape(void) {
    vector<int> filter_shape{ this->filter_height, this->filter_width };
    return filter_shape;
}
vector< vector<MatrixXf> > LCN_Layer::get_activated(void) {
    return this->_activated;
}
vector<vector <MatrixXf> > LCN_Layer::get_delta(void) {
    return this->delta;
}


#endif // INCLUDE_local_constrast_normalization_layer_h_
