#include <NN/model.h>
#include <NN/layer.h>
#include <Func/std_func.h>
#include <LinAlg/tensor.h>
#include <Rand/random.h>
#include <NN/std_opt.h>
#include <gtest/gtest.h>

TEST(Model, AddLayerSizeMismatchThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> first_layer {2, 2};
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> second_layer {3, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(first_layer);

    EXPECT_THROW(model.add_layer(second_layer), std::invalid_argument);
}

TEST(Model, AddLayerAfterInitThrows) {    
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> first_layer {2, 2};
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> second_layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(first_layer);

    model.init();

    EXPECT_THROW(model.add_layer(second_layer), std::invalid_argument);
}

TEST(Model, InitEmptyModelThrows) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt}; 
    
    EXPECT_THROW(model.init(), std::invalid_argument);
}

TEST(Model, ForwardBeforeInitThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);

    LinAlg::Tensor<float> X {{1, 2}, 1};

    EXPECT_THROW(model.forward_pass(X), std::invalid_argument);
}

TEST(Model, ForwardWrongRankThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2}, 1};

    EXPECT_THROW(model.forward_pass(X), std::invalid_argument);

    LinAlg::Tensor<float> Y {{1, 1, 2}, 1};

    EXPECT_THROW(model.forward_pass(Y), std::invalid_argument);
}

TEST(Model, ForwardWrongShapeThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2, 2}, 1};

    EXPECT_THROW(model.forward_pass(X), std::invalid_argument);

    LinAlg::Tensor<float> Y {{1, 3}, 1};

    EXPECT_THROW(model.forward_pass(Y), std::invalid_argument);
}

TEST(Model, CalculateLossWrongTargetRankThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 1};
    model.forward_pass(X);

    LinAlg::Tensor<float> target_1 {{2}, 1};

    EXPECT_THROW(model.calculate_loss(target_1), std::invalid_argument);

    LinAlg::Tensor<float> target_2 {{1, 1, 2}, 1};

    EXPECT_THROW(model.calculate_loss(target_2), std::invalid_argument);
}

TEST(Model, CalculateLossWrongTargteShapeThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 1};
    model.forward_pass(X);

    LinAlg::Tensor<float> target_1{{2, 2}, 1};

    EXPECT_THROW(model.calculate_loss(target_1), std::invalid_argument);

    LinAlg::Tensor<float> target_2 {{1, 3}, 1};

    EXPECT_THROW(model.calculate_loss(target_2), std::invalid_argument);
}

TEST(Model, BackpropagationWrongTargetRankThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 1};
    model.forward_pass(X);

    LinAlg::Tensor<float> target_1 {{2}, 1};

    EXPECT_THROW(model.backpropagation(target_1), std::invalid_argument);

    LinAlg::Tensor<float> target_2 {{1, 1, 2}, 1};

    EXPECT_THROW(model.backpropagation(target_2), std::invalid_argument);
}

TEST(Model, BackpropagationWrongTargetShapeThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 1};
    model.forward_pass(X);

    LinAlg::Tensor<float> target_1 {{2, 2}, 1};

    EXPECT_THROW(model.backpropagation(target_1), std::invalid_argument);

    LinAlg::Tensor<float> target_2 {{1, 3}, 1};

    EXPECT_THROW(model.backpropagation(target_2), std::invalid_argument);
}

TEST(Model, ForwardComposition) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
    
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer_1 {2, 2};
    layer_1.weights.value[{0, 0}] = 1;
    layer_1.weights.value[{0, 1}] = 1;
    layer_1.weights.value[{0, 2}] = 0;
    layer_1.weights.value[{1, 0}] = 1;
    layer_1.weights.value[{1, 1}] = -1;
    layer_1.weights.value[{1, 2}] = 0;

    NN::Layer<float, Func::Linear<float>, Func::No_Activation<float>> layer_2 {2, 1};
    layer_2.weights.value[{0, 0}] = 2;
    layer_2.weights.value[{0, 1}] = 3;
    layer_2.weights.value[{0, 2}] = 1;

    model.add_layer(layer_1);
    model.add_layer(layer_2);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}};
    X[{0, 0}] = 2;
    X[{0, 1}] = 3;

    LinAlg::Tensor<float> out {model.forward_pass(X)};

    LinAlg::Tensor<float> out_exp {{1, 1}};
    out_exp[{0, 0}] = 11;

    EXPECT_EQ(out, out_exp);
}

TEST(Model, ZeroGrad) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
     
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
    Rand::Random<float> random {42};

    layer.normal(random, 0, 1);

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 3};

    model.forward_pass(X);

    LinAlg::Tensor<float> target {{1, 2}, 9};

    model.backpropagation(target);

    model.zero_grad();

    const std::vector<NN::Parameter<float>*> params {model.get_parameters()};

    LinAlg::Tensor<float> grad_exp {{2, 3}};

    EXPECT_EQ(grad_exp, params[0]->grad);
}

TEST(Model, WeightsUpdateWithStep) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
     
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
    Rand::Random<float> random {42};

    layer.normal(random, 0, 1);

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 3};

    model.forward_pass(X);

    LinAlg::Tensor<float> target {{1, 2}, 9};

    model.calculate_loss(target);
    model.backpropagation(target);

    std::vector<NN::Parameter<float>*> pparams {model.get_parameters()};

    LinAlg::Tensor<float> before {pparams[0]->value.copy()};
    
    model.optimizer_step();

    EXPECT_NE(before, model.get_parameters()[0]->value);
}

TEST(Model, SingleTrainStepReducesLoss) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
     
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
    Rand::Random<float> random {42};

    layer.normal(random, 0, 1);

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{1, 2}, 3};

    model.forward_pass(X);

    LinAlg::Tensor<float> target {{1, 2}, 9};

    float loss {model.calculate_loss(target)};
    model.backpropagation(target);
    model.optimizer_step();

    model.forward_pass(X);

    EXPECT_LT(model.calculate_loss(target), loss);
}

TEST(Model, TrainLoopReturnsPerEpochLosses) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
     
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
    Rand::Random<float> random {42};

    layer.normal(random, 0, 1);

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> input {{5, 2}, 1};
    LinAlg::Tensor<float> target {{5, 2}, 9};

    std::vector<float> losses {model.train_loop(input, target, 20)};

    EXPECT_EQ(static_cast<int>(losses.size()), 20);

    EXPECT_LT(losses.back(), losses[0]);
}

TEST(Model, TestLoopDoesNotChangeWeights) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.1f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
     
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
    Rand::Random<float> random {42};

    layer.normal(random, 0, 1);

    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> input {{5, 2}, 1};
    LinAlg::Tensor<float> target {{5, 2}, 9};

    LinAlg::Tensor<float> before {model.get_parameters()[0]->value.copy()};

    model.test_loop(input, target);

    EXPECT_EQ(before, model.get_parameters()[0]->value);
}

TEST(Model, Deterministic) {
    std::vector<std::vector<float>> losses {};

    for(int i {}; i < 2; ++i) {
        Func::MSE<float> loss_fn {};
        NN::Gradient_descent<float> opt {0.1f};

        NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
        
        NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 2};
        Rand::Random<float> random {42};

        layer.normal(random, 0, 1);

        model.add_layer(layer);
        model.init();

        LinAlg::Tensor<float> input {{5, 2}, 1};
        LinAlg::Tensor<float> target {{5, 2}, 9};

        losses.push_back(model.train_loop(input, target, 10));
    }

    for(int i {}; i < 10; ++i) {
        EXPECT_EQ(losses[0][i], losses[1][i]);
    }
}