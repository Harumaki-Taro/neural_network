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
    virtual void update(Loss &loss, Neural_Network& nn);

private:
    float learning_rate;
};


SGD::SGD(float learning_rate) {
    this->learning_rate = learning_rate;
}


void SGD::update(Loss &loss, Neural_Network& nn) {
    for ( int i = 0; i != (int)loss.get_nn_index().size(); i++ ) {
        int j = loss.get_nn_index()[i];
        if ( nn.get_layers()[j]->get_type() == "input_layer" ) {
            ;
        } else if ( nn.get_layers()[j]->get_type() == "full_connect_layer" ) {
            for ( int k = 0; k < nn.get_layers()[j]->_dE_dbW.size(); k++ ) {
                for ( int l = 0; l < nn.get_layers()[j]->_dE_dbW[0].size(); l++ ) {
                    MatrixXf tmp = nn.get_layers()[j]->_dE_dbW[k][l];

                    for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                        if ( loss.get_terms()[m].name == "Lp_norm" ) {
                            if ( loss.get_terms()[m].ord == 2 ) {
                                tmp += loss.get_terms()[m].eps * nn.get_layers()[j]->get_bW()[k][l];
                            } else {
                                cout << "実装されていません。" << endl;
                                exit(1);
                            }
                        }
                    }
                    nn.get_layers()[j]->bW[k][l]
                        = nn.get_layers()[j]->bW[k][l] - this->learning_rate * tmp;
                }
            }
        } else if ( nn.get_layers()[j]->get_type() == "flatten_layer" ) {
            ;
        } else if ( nn.get_layers()[j]->get_type() == "en_tensor_layer" ) {
            ;
        } else if ( nn.get_layers()[j]->get_type() == "convolution_layer" ) {
            for ( int k = 0; k < nn.get_layers()[j]->_dE_dbW.size(); k++ ) {
                for ( int l = 0; l < nn.get_layers()[j]->_dE_dbW[0].size(); l++ ) {
                    MatrixXf tmp = nn.get_layers()[j]->dE_dW[k][l];

                    for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                        if ( loss.get_terms()[m].name == "Lp_norm" ) {
                            if ( loss.get_terms()[m].ord == 2 ) {
                                tmp += loss.get_terms()[m].eps * nn.get_layers()[j]->get_W()[k][l];
                            } else {
                                cout << "実装されていません。" << endl;
                                exit(1);
                            }
                        }
                    }
                    nn.get_layers()[j]->W[k][l]
                        = nn.get_layers()[j]->W[k][l] - this->learning_rate * tmp;
                }
            }
            MatrixXf tmp = nn.get_layers()[j]->dE_db;
            for ( int m = 0; m < loss.get_terms().size(); m++ ) {
                if ( loss.get_terms()[m].name == "Lp_norm" ) {
                    if ( loss.get_terms()[m].ord == 2 ) {
                        tmp += loss.get_terms()[m].eps * nn.get_layers()[j]->get_b();
                    } else {
                        cout << "実装されていません。" << endl;
                        exit(1);
                    }
                }
            }
            nn.get_layers()[j]->b
                = nn.get_layers()[j]->b - this->learning_rate * tmp;
        } else if ( nn.get_layers()[j]->get_type() == "max_pooling_layer" ) {
            ;
        } else if ( nn.get_layers()[j]->get_type() == "output_layer" ) {
            ;
        } else {
            cout << "実装されていません" << endl;
            exit(1);
        }
    }
}


#endif // INCLUDE_sgd_h_
