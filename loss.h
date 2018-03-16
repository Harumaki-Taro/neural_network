#ifndef INCLUDE_loss_h_
#define INCLUDE_loss_h_

#include <iostream>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "neural_network.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using Eigen::MatrixXf;


class Term {
public:
    Term(string name, float eps, int ord, vector<int> index)
        : name(name), eps(eps), ord(ord) , index(index){ }
    string name;
    float eps;
    int ord;
    vector<int> index;
};


class Loss {
public:
    Loss();
    Loss(Neural_Network& nn, vector<int> index={});
    void add_LpNorm(float eps, int ord=2, vector<int> index={});

    vector<int> get_nn_index(void);
    vector<Term> get_terms(void);

private:
    vector<Term> terms;
    vector<int> nn_index;
    int nn_layers_num;
};


Loss::Loss() { ; }


Loss::Loss(Neural_Network& nn, vector<int> index) {
    this->nn_layers_num = nn.get_layers().size();
    if ( index.size() == 0 ) {
        for ( int i = 0; i < nn.get_layers().size(); i++ ) {
            index.push_back(i);
        }
    }
    this->nn_index = index;
}


void Loss::add_LpNorm(float eps, int ord, vector<int> index) {
    if ( index.size() == 0 ) {
        for ( int i = 0; i < this->nn_layers_num; i++ ) {
            index.push_back(i);
        }
    }

    Term term("Lp_norm", eps, ord, index);
    this->terms.push_back(term);
}


vector<int> Loss::get_nn_index(void) { return this->nn_index; }
vector<Term> Loss::get_terms(void) { return this->terms; }


#endif // INCLUDE_loss_h_
