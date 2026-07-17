#include <NN/parameter.h>
#include <NN/std_opt.h>
#include <LinAlg/tensor.h>
#include <vector>
#include <cassert>

void gradient_descent_test() {
    NN::Parameter<float> param {{1, 3}, 1};
    param.value[{0, 1}] = 5;
    param.value[{0, 2}] = -2;

    param.grad[{0, 0}] = 8;
    param.grad[{0, 2}] = -5;

    NN::Gradient_descent<float> gd {2};

    std::vector<NN::Parameter<float>*> params;
    params.push_back(&param);

    gd.init(params);
    gd.step();

    LinAlg::Tensor<float> val_exp {{1, 3}, -15};
    val_exp[{0, 1}] = 5;
    val_exp[{0, 2}] = 8;

    assert(param.value == val_exp && "Gradient descent step fails");
}

int main() {
    gradient_descent_test();

    return 0;
}