#ifndef INCLUDE_batch_normalization_layer_h_
#define INCLUDE_batch_normalization_layer_h_

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

// NOTE:gammaとbetaを学習可能にする
class Batch_Norm_Layer : public Layer {
public:
    virtual void forwardprop(const vector< vector<MatrixXf> > X);
    virtual void calc_delta(const shared_ptr<Layer> &next_layer,
                            const shared_ptr<Layer> &prev_layer);
    virtual void allocate_memory(const int batch_size,
                                 const shared_ptr<Layer> &prev_layer);

    Batch_Norm_Layer(void);
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
    const string type = "batch_normalization_layer";
    bool is_tensor;
    int unit_num;
    vector<int> X_shape;
    int batch_size;
    int channel_num;
    float eps = 0.000001;
    vector<MatrixXf> mu;
    vector<MatrixXf> var;
    vector< vector<MatrixXf> > z;
    vector<MatrixXf> gamma;
    vector<MatrixXf> beta;
    vector< vector<MatrixXf> > d_z;
    vector<MatrixXf> d_mu;
    vector<MatrixXf> d_var;
    vector<MatrixXf> d_var_d_mu;
    vector<MatrixXf> d_gamma;
    vector<MatrixXf> d_beta;
};


Batch_Norm_Layer::Batch_Norm_Layer(void) {
}


void Batch_Norm_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    if ( this->is_tensor ) {
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int h = 0; h < this->X_shape[2]; ++h ) {
                for ( int w = 0; w < this->X_shape[3]; ++w ) {
                    // mu
                    this->mu[c](h, w) = 0.f;
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->mu[c](h, w) += X[n][c](h, w);
                    }
                    this->mu[c](h, w) /= (float)this->batch_size;

                    // var
                    this->var[c](h, w) = 0.f;
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->var[c](h, w) += powf(X[n][c](h, w) - this->mu[c](h, w), 2.f);
                    }
                    this->var[c](h, w) /= (float)this->batch_size;

                    // z, activate
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->z[n][c](h, w) = (X[n][c](h, w) - this->mu[c](h, w))
                                            / sqrt(this->var[c](h, w) + this->eps);
                        this->_activated[n][c](h, w) = this->gamma[c](h, w) * this->z[n][c](h, w) + this->beta[c](h, w);
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int p = 0; p < this->X_shape[3]; ++p ) {
                // mu
                this->mu[0](c, p) = 0.f;
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->mu[0](c, p) += X[0][c](n, p);
                }
                this->mu[0](c, p) /= (float)this->batch_size;

                // var
                this->var[0](c, p) = 0.f;
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->var[0](c, p) += powf(X[0][c](n, p) - this->mu[0](c, p), 2.f);
                }
                this->var[0](c, p) /= (float)this->batch_size;

                // z, activate
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->z[0][c](n, p) = (X[0][c](n, p) - this->mu[0](c, p))
                                        / sqrt(this->var[0](c, p) + this->eps);
                    this->_activated[0][c](n, p) = this->gamma[0](c, p) * this->z[0][c](n, p) + this->beta[0](c, p);
                }
            }
        }
    }
}


void Batch_Norm_Layer::calc_delta(const shared_ptr<Layer> &next_layer,
                                  const shared_ptr<Layer> &prev_layer) {
    if ( this->is_tensor ) {
        // d_gamma, d_beta
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int h = 0; h < this->X_shape[2]; ++h ) {
                for ( int w = 0; w < this->X_shape[3]; ++w ) {
                    this->d_gamma[c](h, w) = 0.f;
                    this->d_beta[c](h, w) = 0.f;
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->d_gamma[c](h, w) += next_layer->delta[n][c](h, w) * this->z[n][c](h, w);
                        this->d_beta[c](h, w) += next_layer->delta[n][c](h, w);
                    }
                }
            }
        }

        // delta
        #pragma omp parallel for
        for ( int n = 0; n < this->batch_size; ++n ) {
            for ( int c = 0; c < this->channel_num; ++c ) {
                for ( int h = 0; h < this->X_shape[2]; ++h ) {
                    for ( int w = 0; w < this->X_shape[3]; ++w ) {
                        this->delta[n][c](h, w) = next_layer->delta[n][c](h, w)
                                                - (this->d_beta[c](h, w) + this->z[n][c](h, w) * this->d_gamma[c](h, w))
                                                / (float)this->batch_size;
                        this->delta[n][c](h, w) *= this->gamma[c](h, w) / sqrt(this->var[c](h, w) + this->eps);
                    }
                }
            }
        }
    } else {
        // d_gamma, d_beta
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int p = 0; p < this->X_shape[3]; ++p ) {
                this->d_gamma[0](c, p) = 0.f;
                this->d_beta[0](c, p) = 0.f;
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->d_gamma[0](c, p) += next_layer->delta[0][c](n, p) * this->z[0][c](n, p);
                    this->d_beta[0](c, p) += next_layer->delta[0][c](n, p);
                }
            }
        }

        // delta
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int n = 0; n < this->batch_size; ++n ) {
                for ( int p = 0; p < this->X_shape[3]; ++p ) {
                    this->delta[0][c](n, p) = next_layer->delta[0][c](n, p)
                                            - (this->d_beta[0](c, p) + this->z[0][c](n, p) * this->d_gamma[0](c, p))
                                            / (float)this->batch_size;
                    this->delta[0][c](n, p) *= this->gamma[0](c, p) / sqrt(this->var[0](c, p) + this->eps);
                }
            }
        }
    }
}


void Batch_Norm_Layer::allocate_memory(const int batch_size,
                                       const shared_ptr<Layer> &prev_layer) {
    this->batch_size = batch_size;
    this->is_tensor = prev_layer->get_is_tensor();
    this->channel_num = prev_layer->get_channel_num();
    this->X_shape.resize(4);

    if ( this->is_tensor ) {
        this->X_shape[0] = this->batch_size;
        this->X_shape[1] = prev_layer->get_channel_num();
        this->X_shape[2] = prev_layer->get_output_map_shape()[0];
        this->X_shape[3] = prev_layer->get_output_map_shape()[1];
        this->unit_num = this->X_shape[1] * this->X_shape[2] * this->X_shape[3];
    } else {
        this->X_shape[0] = 1;
        this->X_shape[1] = prev_layer->get_channel_num();
        this->X_shape[2] = this->batch_size;
        this->X_shape[3] = prev_layer->_activated[0][0].cols();
        this->unit_num = this->X_shape[1] * this->X_shape[3];
    }

    // activated
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_activated;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_activated.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->_activated.push_back(tmp_activated);
    }

    // delta
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_delta;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_delta.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->delta.push_back(tmp_delta);
    }

    // mu
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; c++ ) {
            this->mu.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->mu.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // var
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; c++ ) {
            this->var.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->var.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // z
    for ( int i = 0; i < this->X_shape[0]; i++ ) {
        vector<MatrixXf> tmp_z;
        for ( int j = 0; j < this->X_shape[1]; j++ ) {
            tmp_z.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
        this->z.push_back(tmp_z);
    }

    // gamma
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->gamma.push_back(MatrixXf::Ones(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->gamma.push_back(MatrixXf::Ones(this->channel_num, this->X_shape[3]));
    }

    // beta
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->beta.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->beta.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // d_z
    if ( this->is_tensor ) {
        for ( int n = 0; n < this->batch_size; ++n ) {
            vector<MatrixXf> tmp;
            for ( int c = 0; c < this->channel_num; ++c ) {
                tmp.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
            }
            this->d_z.push_back(tmp);
        }
    } else {
        this->d_z.resize(1);
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_z[0].push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    }

    // d_mu
    if ( this->is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_mu.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->d_mu.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // d_var
    if ( this->is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_var.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->d_var.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // d_var_d_mu
    if ( this->is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_var_d_mu.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->d_var_d_mu.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }

    // d_gamma
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_gamma.push_back(MatrixXf::Ones(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->d_gamma.push_back(MatrixXf::Ones(this->channel_num, this->X_shape[3]));
    }

    // d_beta
    if ( is_tensor ) {
        for ( int c = 0; c < this->channel_num; ++c ) {
            this->d_beta.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
        }
    } else {
        this->d_beta.push_back(MatrixXf::Zero(this->channel_num, this->X_shape[3]));
    }
}


bool Batch_Norm_Layer::get_trainable(void) { return this->trainable; }
string Batch_Norm_Layer::get_type(void) { return this->type; }
bool Batch_Norm_Layer::get_is_tensor(void) { return this->is_tensor; }
int Batch_Norm_Layer::get_unit_num(void) { return this->unit_num; }
int Batch_Norm_Layer::get_batch_size(void) { return this->batch_size; }
int Batch_Norm_Layer::get_channel_num(void) { return this->channel_num; }
vector<int> Batch_Norm_Layer::get_input_map_shape(void) {
    vector<int> input_map_shape;
    if ( this->is_tensor ) {
        input_map_shape = { this->X_shape[2], this->X_shape[3] };
    } else {
        cout << "この活性化層の直前の層はtensorではありません" << endl;
        exit(1);
    }

    return input_map_shape;
}
vector<int> Batch_Norm_Layer::get_output_map_shape(void) {
    return this->get_input_map_shape();
}
vector< vector<MatrixXf> > Batch_Norm_Layer::get_activated(void) {
    return this->_activated;
}
vector<vector <MatrixXf> > Batch_Norm_Layer::get_delta(void) {
    return this->delta;
}


#endif // INCLUDE_batch_normalization_layer_h_
