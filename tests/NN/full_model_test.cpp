#include <LinAlg/tensor.h>
#include <NN/model.h>
#include <NN/layer.h>
#include <Func/std_func.h>
#include <NN/std_opt.h>
#include <Rand/random.h>
#include <cmath>
#include <gtest/gtest.h>

TEST(FullModel, LinearRegression) {
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer {1, 1};
    Rand::Random<float> random {42};
    layer.normal(random, 0, 1);

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> inputs {{20, 1}};
    LinAlg::Tensor<float> targets {{20, 1}};

    for(int i {}; i < 20; ++i) {
        inputs[{i, 0}] = i;
        targets[{i, 0}] = 2 * i + 1;
    }

    model.train_loop(inputs, targets, random, 200, 2);

    LinAlg::Tensor<float> w_exp {{1, 2}};
    w_exp[{0, 0}] = 2;
    w_exp[{0, 1}] = 1;

    EXPECT_TRUE(LinAlg::all_close<float>(w_exp, model.get_parameters()[0]->value, 1e-4, 1e-4));
}

TEST(FullModel, ConvergenceXOR) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer_1 {2, 4};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_2 {4, 2};
    Rand::Random<float> random {42};

    layer_1.normal(random, 0, 1);
    layer_2.normal(random, 0, 1);

    LinAlg::Tensor<float> inputs {{4, 2}};
    inputs[{1, 0}] = 1;
    inputs[{2, 1}] = 1;
    inputs[{3, 0}] = 1;
    inputs[{3, 1}] = 1;

    LinAlg::Tensor<float> targets {{4, 2}};
    targets[{0, 0}] = 1;
    targets[{1, 1}] = 1;
    targets[{2, 1}] = 1;
    targets[{3, 0}] = 1;

    Func::Softmax_cross_entropy<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::Softmax_cross_entropy<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer_1);
    model.add_layer(layer_2);
    model.init();

    std::vector<float> losses = model.train_loop(inputs, targets, random, 500, 1);
    EXPECT_LT(losses.back(), 0.05f);

    for(int i {}; i < 4; ++i) {
        int pred {model.forward_pass(inputs.row(i).unsqueeze()).argmax()[1]};

        int expected {targets.row(i).unsqueeze().argmax()[1]};

        EXPECT_EQ(pred, expected);
    }
}