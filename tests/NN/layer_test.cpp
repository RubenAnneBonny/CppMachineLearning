#include <NN/layer.h>
#include <Func/std_func.h>
#include <LinAlg/tensor.h>
#include <Debug/gradient_check.h>
#include <gtest/gtest.h>

TEST(Layer, WeightInitSeeded) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> a {2, 3};
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> b {2, 3};

    Rand::Random<float> r_1 {42};
    Rand::Random<float> r_2 {42};

    a.normal(r_1, 0, 1);
    b.normal(r_2, 0, 1);

    EXPECT_EQ(a.weights.value, b.weights.value);

    EXPECT_NE(a.weights.value, LinAlg::Tensor<float>({3, 3}, 0));
}

TEST(Layer, ForwardPass) {
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

    EXPECT_EQ(Y, Y_exp);
}

TEST(Layer, BackwardPass) {
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

    EXPECT_EQ(layer.weights.grad, dW_exp);

    LinAlg::Tensor<float> dX_exp {{1, 2}, 0};
    dX_exp[{0, 0}] = 19;
    dX_exp[{0, 1}] = 15;

    EXPECT_EQ(dX, dX_exp);
}

TEST(Layer, GradientAccumulates) {
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

    layer.backward_pass(dY);
    layer.backward_pass(dY);

    LinAlg::Tensor<float> dW_exp {{3, 3}, 0};
    dW_exp[{1, 0}] = -4;
    dW_exp[{1, 1}] = 8;
    dW_exp[{1, 2}] = 4;
    dW_exp[{2, 0}] = -6;
    dW_exp[{2, 1}] = 12;
    dW_exp[{2, 2}] = 6;

    EXPECT_EQ(layer.weights.grad, dW_exp);
}

TEST(Layer, NoActivationForwardBackward) {
    NN::Layer<float, Func::Linear<float>, Func::No_Activation<float>> layer {2, 2};
    layer.weights.value[{0, 0}] = 1;
    layer.weights.value[{0, 1}] = 2;
    layer.weights.value[{0, 2}] = 3;
    layer.weights.value[{1, 0}] = -1;
    layer.weights.value[{1, 1}] = 1;
    layer.weights.value[{1, 2}] = 0;

    LinAlg::Tensor<float> X {{1, 2}};
    X[{0, 0}] = 2;
    X[{0, 1}] = -1;

    LinAlg::Tensor<float> Y {layer.forward_pass(X)};
    LinAlg::Tensor<float> Y_exp {{1, 2}};
    Y_exp[{0, 0}] = 3;
    Y_exp[{0, 1}] = -3;

    EXPECT_EQ(Y, Y_exp);

    LinAlg::Tensor<float> dY {{1, 2}};
    dY[{0, 0}] = 1;
    dY[{0, 1}] = 2;
    LinAlg::Tensor<float> dX {layer.backward_pass(dY)};

    LinAlg::Tensor<float> dX_exp {{1, 2}};
    dX_exp[{0, 0}] = -1;
    dX_exp[{0, 1}] = 4;
    EXPECT_EQ(dX, dX_exp);

    LinAlg::Tensor<float> dW_exp {{2, 3}};
    dW_exp[{0, 0}] = 2;
    dW_exp[{0, 1}] = -1;
    dW_exp[{0, 2}] = 1;
    dW_exp[{1, 0}] = 4;
    dW_exp[{1, 1}] = -2;
    dW_exp[{1, 2}] = 2;

    EXPECT_EQ(layer.weights.grad, dW_exp);
}

namespace {
    template <std::floating_point T>
    class Square {
        public:
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                return LinAlg::pairwise_mult<T>(X, X);
            }

            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                return 2 * X;
            }
    };

    template <typename L>
    double layer_grad_check(L& layer, LinAlg::Tensor<double>& X, int nodes, double eps = 1e-5) {
        auto eval {
            [&]()
            {
                return layer.forward_pass(X).sum();
            }
        };

        layer.forward_pass(X);
        LinAlg::Tensor<double> dY {{1, nodes}, 1};
        layer.weights.zero_grad();
        LinAlg::Tensor<double> analytic_dX {layer.backward_pass(dY)};

        double worst {};

        {
            auto [error, index] = Debug::GradCheck::Helper::max_rel_error(X, analytic_dX, eval, eps);
            worst = (worst > error ? worst : error);
        }

        for(int i {}; i < nodes; ++i) {
            LinAlg::Tensor<double> w_row {layer.weights.value.row(i).unsqueeze()};
            LinAlg::Tensor<double> grad_row {layer.weights.grad.row(i).unsqueeze()};
            auto [error, index] = Debug::GradCheck::Helper::max_rel_error(w_row, grad_row, eval, eps);
            worst = (worst > error ? worst : error);
        }

        return worst;
    }
}

TEST(Layer, NumericalGradientCheckNoActivation) {
    NN::Layer<double, Func::Linear<double>, Func::No_Activation<double>> layer {3, 2};
    Rand::Random<double> random {42};
    layer.normal(random, 0, 1);

    LinAlg::Tensor<double> X {{1, 3}};
    X.normal(random, 0, 1);

    EXPECT_LT(layer_grad_check(layer, X, 2), 1e-6);
}

TEST(Layer, NumricalGradientCheckSmoothActivation) {
    NN::Layer<double, Func::Linear<double>, Square<double>> layer {3, 2};
    Rand::Random<double> random {42};
    layer.normal(random, 0, 1);

    LinAlg::Tensor<double> X {{1, 3}};
    X.normal(random, 0, 1);

    EXPECT_LT(layer_grad_check(layer, X, 2), 1e-6);
}