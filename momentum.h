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
    Momentum(float learning_rate, float mu=0.5);
    virtual void update(Loss &loss, Neural_Network& nn, int step);

private:
    float learning_rate;
    float mu;
    vector< vector< vector<MatrixXf> > >accumulation_W;
    vector<MatrixXf> accumulation_b;
};


Momentum::Momentum(float learning_rate, float mu) {
    this->learning_rate = learning_rate;
    this->mu = mu;
}


void Momentum::update(Loss &loss, Neural_Network& nn, int step) {
    // set accumulation_W, accumulation_b
    if ( step == 0 ) {
        this->accumulation_W.clear();
        this->accumulation_W.shrink_to_fit();
        this->accumulation_b.clear();
        this->accumulation_b.shrink_to_fit();
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
            this->accumulation_W.push_back(tmp_vec);
            this->accumulation_b.push_back(nn.get_layers()[i]->b);
        }
    }

    // update
    MatrixXf prev_tmp;
    MatrixXf grad;
    for ( int i = 0; i != (int)loss.get_nn_index().size(); i++ ) {
        int j = loss.get_nn_index()[i];
        if ( nn.get_layers()[j]->get_trainable() ) {
            for ( int k = 0; k < nn.get_layers()[j]->dE_dW.size(); k++ ) {
                for ( int l = 0; l < nn.get_layers()[j]->dE_dW[0].size(); l++ ) {
                    grad = nn.get_layers()[j]->dE_dW[k][l];

                    for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                        if ( loss.get_terms()[m].name == "Lp_norm" ) {
                            grad += Lp_norm(loss.get_terms()[m], nn.get_layers()[j]->W[k][l], j);
                        }
                    }
                    this->accumulation_W[j][k][l] = this->mu * this->accumulation_W[j][k][l] - this->learning_rate * grad;
                    nn.get_layers()[j]->W[k][l] += this->accumulation_W[j][k][l];
                }
            }
        }
        if ( nn.get_layers()[j]->get_type() == "convolution_layer" ) {
            grad = nn.get_layers()[j]->dE_db;

            for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                if ( loss.get_terms()[m].name == "Lp_norm" ) {
                    grad += Lp_norm(loss.get_terms()[m], nn.get_layers()[j]->b, j);
                }
            }
            this->accumulation_b[j] = this->mu * this->accumulation_b[j] - this->learning_rate * grad;
            nn.get_layers()[j]->b += this->accumulation_b[j];
        }
    }
}


#endif // INCLUDE_momentum_h_
