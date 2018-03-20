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
    virtual void forwardprop(const MatrixXf X);
    virtual void allocate_memory(const int batch_size,
                                 const int example_size);

    // getter
    virtual bool get_trainable(void);
    virtual string get_type(void);
    virtual bool get_is_tensor(void);
    virtual int get_unit_num(void);
    virtual vector<vector <MatrixXf> > get_activated(void);
    virtual int get_batch_size(void);
    int get_example_size(void);

    // setter
    virtual void set_batch_size(const int batch_size);

private:
    bool _trainable = false;
    const string type = "input_layer";
    const bool is_tensor = false;
    int unit_num;
    // Parameters specified at first
    int batch_size;
    int _example_size;
};


void Input_Layer::forwardprop(const MatrixXf X) {
    this->_activated[0][0] = X;
}


void Input_Layer::allocate_memory(const int batch_size, const int example_size) {
    this->batch_size = batch_size;
    this->_example_size = example_size;
    this->unit_num = this->_example_size;

    this->_activated.resize(1); this->_activated[0].resize(1);
    this->_activated[0][0].resize(this->batch_size, this->_example_size);
}


bool Input_Layer::get_trainable(void) { return this->_trainable; }
string Input_Layer::get_type(void) { return this->type; }
bool Input_Layer::get_is_tensor(void) { return this->is_tensor; }
int Input_Layer::get_unit_num(void) { return this->unit_num; }
vector<vector <MatrixXf> > Input_Layer::get_activated(void) { return this->_activated; }
int Input_Layer::get_batch_size(void) { return this->batch_size; }
int Input_Layer::get_example_size(void) { return this->_example_size; }


void Input_Layer::set_batch_size(const int batch_size) {
    this->allocate_memory(batch_size, this->_example_size);
}
