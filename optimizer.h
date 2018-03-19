#ifndef INCLUDE_optimizer_h_
#define INCLUDE_optimizer_h_

#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "my_math.h"
#include "neural_network.h"
#include "loss.h"

using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;


class Optimizer {
public:
    virtual void update(Loss &loss, Neural_Network& nn, int step) { cout << "opt使用禁止" << endl; exit(1); }

};


MatrixXf Lp_norm(Term term, MatrixXf W, int layer_num) {
    MatrixXf output;

    if ( std::find(term.index.begin(), term.index.end(), layer_num) != term.index.end() ) {
        if ( term.ord == 2 ) {
            output = term.eps * W;
        } else {
            cout << "実装されていません。" << endl;
            exit(1);
        }
    }

    return output;
}


#endif // INCLUDE_optimizer_h_
