#include <functional>
#include <math.h>
#include "my_math.h"
#include "layer.h"
#include "Eigen/Core"


using std::function;
using std::cout;
using std::endl;
using Eigen::MatrixXf;

class Input_Layer : public Layer {
public:
    virtual void allocate_memory(int, int, bool);

    // getter
    virtual MatrixXf get_activated_(void);
    virtual int get_batch_size(void);

private:
    // Parameters specified at first
    int batch_size;
    // 今後削除予定。
    MatrixXf W;
};

void Input_Layer::allocate_memory(int batch_size, int example_size, bool use_bias_in_next_layer) {
    this->batch_size = batch_size;
    this->activated_.resize(batch_size, example_size+1);
    if ( use_bias_in_next_layer ) {
        this->activated_.block(0,0,batch_size,1) = MatrixXf::Ones(batch_size, 1);
    } else {
        this->activated_.block(0,0,batch_size,1) = MatrixXf::Zero(batch_size, 1);
    }
    this->W.resize(batch_size, example_size);
}


MatrixXf Input_Layer::get_activated_(void) { return this->activated_; }
int Input_Layer::get_batch_size(void) { return this->batch_size; }
