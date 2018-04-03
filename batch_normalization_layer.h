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
    float d_mu;
    vector< vector<MatrixXf> > d_var;
};


Batch_Norm_Layer::Batch_Norm_Layer(void) {
}


void Batch_Norm_Layer::forwardprop(const vector< vector<MatrixXf> > X) {
    if ( this->is_tensor ) {
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            // mu
            for ( int h = 0; h < this->X_shape[2]; ++h ) {
                for ( int w = 0; w < this->X_shape[3]; ++w ) {
                    this->mu[c](h, w) = 0.f;
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->mu[c](h, w) += X[n][c](h, w);
                    }
                }
            }
            this->mu[c] /= (float)this->batch_size;

            // var
            for ( int h = 0; h < this->X_shape[2]; ++h ) {
                for ( int w = 0; w < this->X_shape[3]; ++w ) {
                    this->var[c](h, w) = 0.f;
                    for ( int n = 0; n < this->batch_size; ++n ) {
                        this->var[c](h, w) += pow(X[n][c](h, w) - this->mu[c](h, w), 2.f);
                    }
                }
            }
            this->var[c] /= (float)this->batch_size;

            // z
            for ( int n = 0; n < this->batch_size; ++n ) {
                this->z[n][c] = ((X[n][c] - this->mu[c]).array() / sqrt(this->var[c].array() + this->eps)).matrix();
            }

            // activate
            for ( int n = 0; n < this->batch_size; ++n ) {
                this->_activated[n][c] = (this->gamma[c].array() * this->z[n][c].array() + this->beta[c].array()).matrix();
            }
        }
    } else {
        // mu
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int p = 0; p < this->X_shape[3]; ++p ) {
                this->mu[0](c, p) = 0.f;
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->mu[0](c, p) += X[0][c](n, p);
                }
            }
        }
        this->mu[0] /= (float)this->batch_size;

        // var
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int p = 0; p < this->X_shape[3]; ++p ) {
                this->var[0](c, p) = 0.f;
                for ( int n = 0; n < this->batch_size; ++n ) {
                    this->var[0](c, p) += pow(X[0][c](n, p) - this->mu[0](c, p), 2.f);
                }
            }
        }
        this->var[0] /= (float)this->batch_size;

        // z
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int n = 0; n < this->batch_size; ++n ) {
                for ( int p = 0; p < this->X_shape[3]; ++p ) {
                    this->z[0][c](n, p) = pow(X[0][c](n, p) - this->mu[0](c, p), 2.f)
                                        / sqrt(var[0](c, p) + this->eps);
                }
            }
        }

        // activate
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int n = 0; n < this->batch_size; ++n ) {
                for ( int p = 0; p < this->X_shape[3]; ++p ) {
                    this->_activated[0][c](n, p) = this->gamma[0](c, p) * this->z[0][c](n, p) + this->beta[0](c, p);
                }
            }
        }
    }
}


void Batch_Norm_Layer::calc_delta(const shared_ptr<Layer> &next_layer,
                                  const shared_ptr<Layer> &prev_layer) {
    if ( this->is_tensor ) {
        // d_var
        #pragma omp parallel for
        for ( int n = 0; n < this->batch_size; ++n ) {
            for ( int c = 0; c < this->channel_num; ++c ) {
                for ( int h = 0; h < this->X_shape[2]; ++h ) {
                    for ( int w = 0; w < this->X_shape[3]; ++w ) {
                        this->d_var[n][c](h, w) = 0.f;
                        for ( int m = 0; m < this->batch_size; ++m ) {
                            this->d_var[n][c](h, w) -= prev_layer->_activated[m][c](h, w) - this->mu[c](h, w);
                        }
                        this->d_var[n][c](h, w) *= 2.f / pow(this->batch_size, 2.f);
                        this->d_var[n][c](h, w) += 2.f / this->batch_size * (prev_layer->_activated[n][c](h, w) - this->mu[c](h, w));
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
                        this->delta[n][c](h, w) = 0.f;
                        for ( int m = 0; m < this->batch_size; ++ m ) {
                            float tmp = 0.f;
                            tmp = d_mu + (prev_layer->_activated[m][c](h, w) - this->mu[c](h, w))
                                / (2.f * (this->var[c](h, w) + this->eps)) * this->d_var[n][c](h, w);
                            this->delta[n][c](h, w) -= next_layer->delta[m][c](h, w) * tmp;
                        }
                        this->delta[n][c](h, w) += next_layer->delta[n][c](h, w);
                        this->delta[n][c](h, w) /= sqrt(this->var[c](h, w) + this->eps);
                    }
                }
            }
        }
    } else {
        // d_var
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int n = 0; n < this->batch_size; ++n ) {
                for ( int p = 0; p < this->X_shape[3]; ++p ) {
                    this->d_var[0][c](n, p) = 0.f;
                    for ( int m = 0; m < this->batch_size; ++m ) {
                        this->d_var[0][c](n, p) -= prev_layer->_activated[0][c](n, p) - this->mu[0](c, p);
                    }
                    this->d_var[0][c](n, p) *= 2.f / pow(this->batch_size, 2.f);
                    this->d_var[0][c](n, p) += 2.f / this->batch_size * (prev_layer->_activated[0][c](n, p) - this->mu[0](c, p));
                }
            }
        }

        // delta
        #pragma omp parallel for
        for ( int c = 0; c < this->channel_num; ++c ) {
            for ( int n = 0; n < this->batch_size; ++n ) {
                for ( int p = 0; p < this->X_shape[3]; ++p ) {
                    this->delta[0][c](n, p) = 0.f;
                    for ( int m = 0; m < this->batch_size; ++ m ) {
                        float tmp = 0.f;
                        tmp = d_mu + (prev_layer->_activated[0][c](m, p) - this->mu[0](c, p))
                            / (2.f * (this->var[0](c, p) + this->eps)) * this->d_var[0][c](n, p);
                        this->delta[0][c](n, p) -= next_layer->delta[0][c](m, p) * tmp;
                    }
                    this->delta[0][c](n, p) += next_layer->delta[0][c](n, p);
                    this->delta[0][c](n, p) /= sqrt(this->var[0](c, p) + this->eps);
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

    // d_mu
    this->d_mu = 1.f / (float)this->batch_size;

    // d_var
    if ( is_tensor ) {
        for ( int n = 0; n < this->batch_size; ++n ) {
            vector<MatrixXf> tmp_d_var;
            for ( int c = 0; c < this->channel_num; ++c ) {
                tmp_d_var.push_back(MatrixXf::Zero(this->X_shape[2], this->X_shape[3]));
            }
            this->d_var.push_back(tmp_d_var);
        }
    } else {
        vector<MatrixXf> tmp_d_var;
        for ( int c = 0; c < this->channel_num; ++c ) {
            tmp_d_var.push_back(MatrixXf::Zero(this->batch_size, this->X_shape[3]));
        }
        this->d_var.push_back(tmp_d_var);
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
