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

    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_2 {2, 1};
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

    std::vector<float> losses {model.train_loop(input, target, random, 20, 1)};

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

        losses.push_back(model.train_loop(input, target, random, 10, 1));
    }

    for(int i {}; i < 10; ++i) {
        EXPECT_EQ(losses[0][i], losses[1][i]);
    }
}

TEST(Model, ForwardCapture) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer_1 {2, 2};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_2 {2, 1};
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer_1);
    model.add_layer(layer_2);
    model.init();

    auto& W0 = model.get_parameters()[0]->value;
    auto& W1 = model.get_parameters()[1]->value;
    W0[{0, 0}] = 1;
    W0[{0, 1}] = 1;
    W0[{0, 2}] = -5;
    W0[{1, 0}] = 2;
    W0[{1, 1}] = 1;
    W0[{1, 2}] = 0;
    W1[{0, 0}] = 1;
    W1[{0, 1}] = 2;
    W1[{0, 2}] = 3;

    LinAlg::Tensor<float> X {{1, 2}};
    X[{0, 0}] = 1;
    X[{0, 1}] = 2;

    std::vector<LinAlg::Tensor<float>> capture {model.forward_capture(X)};

    float l0_n0 {capture[0][{0, 0}]};
    float l0_n1 {capture[0][{0, 1}]};
    EXPECT_NEAR(l0_n0, 0, 1e-6f);
    EXPECT_NEAR(l0_n1, 4, 1e-6f);

    float l1_n0 {capture[1][{0, 0}]};
    EXPECT_NEAR(l1_n0, 11, 1e-6f);
}

TEST(Model, PreActivationCapture) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer_1 {2, 2};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_2 {2, 1};
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer_1);
    model.add_layer(layer_2);
    model.init();

    auto& W0 = model.get_parameters()[0]->value;
    auto& W1 = model.get_parameters()[1]->value;
    W0[{0, 0}] = 1;
    W0[{0, 1}] = 1;
    W0[{0, 2}] = -5;
    W0[{1, 0}] = 2;
    W0[{1, 1}] = 1;
    W0[{1, 2}] = 0;
    W1[{0, 0}] = 1;
    W1[{0, 1}] = 2;
    W1[{0, 2}] = 3;

    LinAlg::Tensor<float> X {{1, 2}};
    X[{0, 0}] = 1;
    X[{0, 1}] = 2;

    std::vector<LinAlg::Tensor<float>> capture {model.pre_activation_capture(X)};

    float l0_n0 {capture[0][{0, 0}]};
    float l0_n1 {capture[0][{0, 1}]};
    EXPECT_NEAR(l0_n0, -2, 1e-6f);
    EXPECT_NEAR(l0_n1, 4, 1e-6f);

    float l1_n0 {capture[1][{0, 0}]};
    EXPECT_NEAR(l1_n0, 11, 1e-6f);    
}

namespace {
    template <std::floating_point T>
    T calculate_stddev(const std::vector<T>& values) {
        int num_elements {static_cast<int>(values.size())};

        T sum {};
        for(T value : values) {
            sum += value;
        }

        T mean {sum / num_elements};

        T variance {};
        for(T value : values) {
            variance += (value - mean) * (value - mean);
        }
        variance /= num_elements;

        T stddev {std::sqrt(variance)};

        return stddev;
    }
}

TEST(Model, RandomInitStddevWithinTol) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer_1 {1, 50};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_2 {50, 50};
    NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> layer_3 {50, 50};

    Rand::Random<float> random {42};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    model.add_layer(layer_1);
    model.add_layer(layer_2);
    model.add_layer(layer_3);

    LinAlg::Tensor<float> ssamples {{100, 1}};
    ssamples.uniform(random, 5, 6);

    model.init(random, ssamples);

    LinAlg::Tensor<float> samples {{100, 1}};
    samples.uniform(random, 5, 6);

    for(int layer {}; layer < 3; ++layer) {
        std::vector<float> values {};
        for(int sample {}; sample < 100; ++sample) {
            LinAlg::Tensor<float> X {samples.row(sample).unsqueeze()};
            std::vector<LinAlg::Tensor<float>> outputs {model.pre_activation_capture(X)};

            X = outputs[layer];

            for(int i {}; i < X.get_extent(1); ++i) {
                values.push_back(X[{0, i}]);
            }
        }
        float stddev = calculate_stddev(values);

        EXPECT_NEAR(stddev, 1.0f, 0.1f);
    }
}

TEST(Model, RandomInitCorrectness) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {64, 128};
    
    Rand::Random<float> random {42};

    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};

    LinAlg::Tensor<float> samples {{200, 64}};
    samples.normal(random, 0, 1);

    model.add_layer(layer);
    model.init(random, samples);
    
    std::vector<float> values {};
    for(int i {}; i < 128; ++i) {
        for(int j {}; j < 65; ++j) {
            values.push_back(model.get_parameters()[0]->value[{i, j}]);
        }
    }

    float stddev {calculate_stddev(values)};

    EXPECT_NEAR(stddev, std::sqrt(1.0f / 64), 0.02f);
}