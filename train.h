#ifndef INCLUDE_train_h_
#define INCLUDE_train_h_

#include <iostream>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "Eigen/Core"
#include "neural_network.h"
#include "loss.h"
#include "sgd.h"
#include "optimizer.h"
#include "batch.h"

using std::function;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::shared_ptr;
using Eigen::MatrixXf;


class Train {
public:
    void update(Neural_Network& nn, Mini_Batch mini_batch, int i);
    Train(Loss loss, SGD sgd);
    // Neural_Network output(void);

    // Neural_Network nn;

private:
    Loss loss;
    shared_ptr<Optimizer> opt;
};


Train::Train(Loss loss, SGD sgd) {
    this->loss = loss;
    this->opt = std::make_shared<SGD>(sgd);
}


void Train::update(Neural_Network& nn, Mini_Batch mini_batch, int i) {
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now(); // 計測開始時間

    MatrixXf pred = nn.forwardprop(mini_batch.example);
    nn.backprop(pred, mini_batch.label);

    MatrixXf before = nn.get_layers()[3]->get_bW()[0][0];
    this->opt->update(this->loss, nn);
    MatrixXf after = nn.get_layers()[3]->get_bW()[0][0];

    end = std::chrono::system_clock::now();  // 計測終了時間

    if ( i % 100 == 0 ) {
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
        cout << "step: " << i << "  " << "loss: " << nn.calc_loss_with_prev_pred(mini_batch.label) << " (" << elapsed << " msec)" << endl;
    }
}


#endif // INCLUDE_optimizer_h_
