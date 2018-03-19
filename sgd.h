#ifndef INCLUDE_sgd_h_
#define INCLUDE_sgd_h_

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


class SGD : public Optimizer {
public:
    float eps = 0.001;

    SGD(float learning_rate);
    virtual void update(Loss &loss, Neural_Network& nn, int step);

private:
    float learning_rate;
};


SGD::SGD(float learning_rate) {
    this->learning_rate = learning_rate;
}


void SGD::update(Loss &loss, Neural_Network& nn, int step) {
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
                    nn.get_layers()[j]->W[k][l]
                        -= this->learning_rate * tmp;
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
            nn.get_layers()[j]->b
                -= this->learning_rate * tmp;
        }
    }
}


#endif // INCLUDE_sgd_h_
