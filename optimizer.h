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
    virtual void update(Loss &loss, Neural_Network& nn) { cout << "opt使用禁止" << endl; exit(1); }

};


#endif // INCLUDE_optimizer_h_
