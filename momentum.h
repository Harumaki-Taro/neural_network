#ifndef INCLUDE_momentum_h_
#define INCLUDE_momentum_h_

#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "my_math.h"
#include "neural_network.h"
#include "loss.h"
#include "optimizer.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class Momentum : public Optimizer {
public:
    float eps = 0.001;

    Momentum(float learning_rate, float mu=0.9);
    virtual void update(Loss &loss, Neural_Network& nn, int step);

private:
    float learning_rate;
    float mu;
    vector< vector< vector<MatrixXf> > >prev_W;
    vector<MatrixXf> prev_b;
};


Momentum::Momentum(float learning_rate, float mu) {
    this->learning_rate = learning_rate;
    this->mu = mu;
}


void Momentum::update(Loss &loss, Neural_Network& nn, int step) {
    // set prev_W, prev_b
    if ( step == 0 ) {
        this->prev_W.clear();
        this->prev_W.shrink_to_fit();
        this->prev_b.clear();
        this->prev_b.shrink_to_fit();
        for ( int i = 0; i < nn.get_layers().size(); ++i ) {
            vector< vector<MatrixXf> > tmp_vec;
            for ( int j = 0; j < nn.get_layers()[i]->W.size(); ++j ) {
                vector<MatrixXf> tmp;
                for ( int k = 0; k < nn.get_layers()[i]->W[j].size(); ++k ) {
                    tmp.push_back(MatrixXf::Zero(nn.get_layers()[i]->W[j][k].rows(),
                                                 nn.get_layers()[i]->W[j][k].cols()));
                }
                tmp_vec.push_back(tmp);
            }
            this->prev_W.push_back(tmp_vec);
            this->prev_b.push_back(nn.get_layers()[i]->b);
        }
    }

    // update
    MatrixXf prev_tmp;
    for ( int i = 0; i != (int)loss.get_nn_index().size(); i++ ) {
        int j = loss.get_nn_index()[i];
        if ( nn.get_layers()[j]->get_trainable() ) {
            for ( int k = 0; k < nn.get_layers()[j]->dE_dW.size(); k++ ) {
                for ( int l = 0; l < nn.get_layers()[j]->dE_dW[0].size(); l++ ) {
                    MatrixXf tmp = nn.get_layers()[j]->dE_dW[k][l];

                    for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                        if ( loss.get_terms()[m].name == "Lp_norm" ) {
                            tmp += Lp_norm(loss.get_terms()[m], nn.get_layers()[j]->W[k][l], j);
                        }
                    }
                    prev_tmp = nn.get_layers()[j]->W[k][l];
                    nn.get_layers()[j]->W[k][l] += this->mu * (nn.get_layers()[j]->W[k][l] - this->prev_W[j][k][l]);
                    nn.get_layers()[j]->W[k][l] -= this->learning_rate * tmp;
                    this->prev_W[j][k][l] = prev_tmp;
                }
            }
        }
        if ( nn.get_layers()[j]->get_type() == "convolution_layer" ) {
            MatrixXf tmp = nn.get_layers()[j]->dE_db;

            for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                if ( loss.get_terms()[m].name == "Lp_norm" ) {
                    tmp += Lp_norm(loss.get_terms()[m], nn.get_layers()[j]->b, j);
                }
            }
            prev_tmp = nn.get_layers()[j]->b;
            nn.get_layers()[j]->b += this->mu * (nn.get_layers()[j]->b - this->prev_b[j]);
            nn.get_layers()[j]->b -= this->learning_rate * tmp;
            this->prev_b[j] = prev_tmp;
        }
    }
}


#endif // INCLUDE_momentum_h_
