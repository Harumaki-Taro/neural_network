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
    virtual void allocate_memory(const int batch_size,
                                 const int example_size,
                                 const bool use_bias);

    // getter
    virtual bool get_trainable(void);
    virtual vector<vector <MatrixXf> > get_activated(void);
    virtual int get_batch_size(void);
    int get_example_size(void);

    // setter
    virtual void set_batch_size(const int batch_size,
                                const bool use_bias_in_next_layer);

private:
    bool _trainable = false;
    // Parameters specified at first
    int batch_size;
    int _example_size;
};


void Input_Layer::allocate_memory(const int batch_size, const int example_size, const bool use_bias_in_next_layer) {
    this->batch_size = batch_size;
    this->_example_size = example_size;

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->_example_size+1);
    if ( use_bias_in_next_layer ) {
        this->_activated[0][0].block(0,0,this->batch_size,1) = MatrixXf::Ones(this->batch_size, 1);
    } else {
        this->_activated[0][0].block(0,0,this->batch_size,1) = MatrixXf::Zero(this->batch_size, 1);
    }
}


bool Input_Layer::get_trainable(void) { return this->_trainable; }
vector<vector <MatrixXf> > Input_Layer::get_activated(void) { return this->_activated; }
int Input_Layer::get_batch_size(void) { return this->batch_size; }
int Input_Layer::get_example_size(void) { return this->_example_size; }


void Input_Layer::set_batch_size(const int batch_size, const bool use_bias_in_next_layer) {
    this->allocate_memory(batch_size, this->_example_size, use_bias_in_next_layer);
}
