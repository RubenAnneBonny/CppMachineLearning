#include <cassert>
#include <NN/layer.h>
#include <Func/std_func.h>
#include <LinAlg/tensor.h>

void forward_pass_test() {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3, 3};
    layer.weights.value[{0, 0}] = 2;
    layer.weights.value[{0, 1}] = -4;
    layer.weights.value[{2, 2}] = 8;

    LinAlg::Tensor<float> X {{1, 2}, 2};
    X[{0, 0}] = -1;

    LinAlg::Tensor<float> Y {layer.forward_pass(X)};
    LinAlg::Tensor<float> Y_exp {{1, 3}};
    Y_exp[{0, 1}] = 6;
    Y_exp[{0, 2}] = 11;

    assert(Y == Y_exp && "forward pass through one layer is faulty");
}

void backward_pass_test() {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3, 3};
    layer.weights.value[{0, 0}] = 2;
    layer.weights.value[{0, 1}] = -4;
    layer.weights.value[{1, 0}] = 5;
    layer.weights.value[{2, 2}] = 8;

    LinAlg::Tensor<float> X {{1, 2}, 2};
    X[{0, 0}] = -1;

    layer.forward_pass(X);

    LinAlg::Tensor<float> dY {{1, 3}, 1};
    dY[{0, 1}] = 2;
    dY[{0, 2}] = 3;

    LinAlg::Tensor<float> dX {layer.backward_pass(dY)};

    LinAlg::Tensor<float> dW_exp {{3, 3}, 0};
    dW_exp[{1, 0}] = -2;
    dW_exp[{1, 1}] = 4;
    dW_exp[{1, 2}] = 2;
    dW_exp[{2, 0}] = -3;
    dW_exp[{2, 1}] = 6;
    dW_exp[{2, 2}] = 3;

    assert(layer.weights.grad == dW_exp && "weight gradients are tracked faulty");

    LinAlg::Tensor<float> dX_exp {{1, 2}, 0};
    dX_exp[{0, 0}] = 19;
    dX_exp[{0, 1}] = 15;

    assert(dX == dX_exp && "input gradients are faulty");
}

int main() {
    forward_pass_test();
    backward_pass_test();

    return 0;
}