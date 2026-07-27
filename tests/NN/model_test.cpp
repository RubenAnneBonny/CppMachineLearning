#include <NN/model.h>
#include <NN/layer.h>
#include <Func/std_func.h>
#include <LinAlg/tensor.h>
#include <Rand/random.h>
#include <NN/std_opt.h>
#include <filesystem>
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

    std::vector<float> losses {model.train_loop(random, input, target, 20, 1)};

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

        losses.push_back(model.train_loop(random, input, target, 10, 1));
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

TEST(Model, AddLayerDeepCopies) {
    NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> layer {2, 3};
    Rand::Random<float> random {42};
    layer.normal(random, 0, 1);
    layer.weights.grad.normal(random, 0, 1);

    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {Func::MSE<float>{}, NN::Gradient_descent<float>{0.1f}};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> before_value {model.get_parameters()[0]->value.copy()};
    LinAlg::Tensor<float> before_grad {model.get_parameters()[0]->grad.copy()};

    layer.normal(random, 0, 1);
    layer.weights.grad.normal(random, 0, 1);

    EXPECT_EQ(model.get_parameters()[0]->value, before_value);
    EXPECT_EQ(model.get_parameters()[0]->grad, before_grad);
}

TEST(Model, SavingLoading) {
    const std::filesystem::path dir {"saving_loading_tests"};
    std::filesystem::create_directories(dir);
    const std::filesystem::path path {dir / "SavingLoading.weights"};

    NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> layer {2, 3};

    NN::Adam<float> opt {};
    Func::MSE<float> loss_fn {};

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};

    Rand::Random<float> random {};

    LinAlg::Tensor<float> samples {{20, 2}};
    samples.normal(random, 0, 1);

    model.add_layer(layer);
    model.init(random, samples);

    model.save_weights(path.string());

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {loss_fn, opt};    

    model_load.add_layer(layer);
    model_load.init(random, samples);

    model_load.load_weights(path.string());

    EXPECT_EQ(model.get_parameters()[0]->value, model_load.get_parameters()[0]->value);

    std::filesystem::remove_all(dir);
}

TEST(Model, SavingToUnopenableThrows) {
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {Func::MSE<float>{}, NN::Adam<float>{}};
    model.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model.init();

    EXPECT_THROW(model.save_weights("nonexistant_directory/Save.weights"), std::invalid_argument);
}

TEST(Model, LoadingNumLayerDiffThrows) {
    const std::filesystem::path dir {"saving_loading_tests"};
    std::filesystem::create_directories(dir);
    const std::filesystem::path path {dir / "LoadingNumLayerDiffThrows.weights"};

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_save {Func::MSE<float>{}, NN::Adam<float>{}};
    model_save.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model_save.init();

    model_save.save_weights(path.string());

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {Func::MSE<float>{}, NN::Adam<float>{}};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {3, 3});
    model_load.init();

    EXPECT_THROW(model_load.load_weights(path.string()), std::invalid_argument);

    std::filesystem::remove_all(dir);
}

TEST(Model, LoadingDiffWeightsShapeThrows) {
    const std::filesystem::path dir {"saving_loading_tests"};
    std::filesystem::create_directories(dir);
    const std::filesystem::path path {dir / "SavingLoading.weights"};

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_save {Func::MSE<float>{}, NN::Adam<float>{}};
    model_save.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model_save.init();

    model_save.save_weights(path.string());

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {Func::MSE<float>{}, NN::Adam<float>{}};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 4});
    model_load.init();

    EXPECT_THROW(model_load.load_weights(path.string()), std::invalid_argument);

    std::filesystem::remove_all(dir);
}

TEST(Model, LoadTruncatedFileThrows) {
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {Func::MSE<float>{}, NN::Adam<float>{}};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 4});
    model_load.init();

    const std::filesystem::path path {"truncated.weights"};
    {
        std::ofstream out {path};
        out << "1\n0 3 3\n1.0 2.0 3.0\n";
    }

    EXPECT_THROW(model_load.load_weights(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(Model, BackpropMathcesNumericGradient) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer {2, 1};
    layer.weights.value[{0, 0}] = 0.5f;
    layer.weights.value[{0, 1}] = -1.0f;
    layer.weights.value[{0, 2}] = 0.25f;
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2, 2}};
    X[{0, 0}] = 1.0f;
    X[{0, 1}] = 2.0f;
    X[{1, 0}] = -1.0f;
    X[{1, 1}] = 0.5f;

    LinAlg::Tensor<float> Y {{2, 1}};
    Y[{0, 0}] = 1.0f;
    Y[{1, 0}] = 0.0f;

    model.forward_pass(X);
    model.backpropagation(Y);

    float eps {1e-2f};
    auto& param {model.get_parameters()[0]};
    for(int j {}; j < 3; ++j) {
        float analytic {param->grad[{0, j}]};
        float original {param->value[{0, j}]};
        param->value[{0, j}] = original + eps;
        model.forward_pass(X);
        float loss_plus {model.calculate_loss(Y)};
        param->value[{0, j}] = original - eps;
        model.forward_pass(X);
        float loss_minus {model.calculate_loss(Y)};
        param->value[{0, j}] = original;
        EXPECT_NEAR(analytic, (loss_plus - loss_minus) / (2 * eps), 1e-2f);
    }
}

TEST(Model, CalculateLossUsesMeanOverBatch) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer {2, 1};
    layer.weights.value[{0, 0}] = 0.5f;
    layer.weights.value[{0, 1}] = -1.0f;
    layer.weights.value[{0, 2}] = 0.25f;
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2, 2}};
    X[{0, 0}] = 1.0f;
    X[{0, 1}] = 2.0f;
    X[{1, 0}] = -1.0f;
    X[{1, 1}] = 0.5f;

    LinAlg::Tensor<float> Y {{2, 1}};
    Y[{0, 0}] = 1.0f;
    Y[{1, 0}] = 0.0f;

    LinAlg::Tensor<float> P {model.forward_pass(X)};

    float d0 {P[{0, 0}] - Y[{0, 0}]};
    float d1 {P[{1, 0}] - Y[{1, 0}]};
    float expected {(d0 * d0 + d1 * d1) / 2.0f};

    EXPECT_NEAR(model.calculate_loss(Y), expected, 1e-5f);
}

TEST(Model, BatchMatchesSingle) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};

    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_single {2, 1};
    layer_single.weights.value[{0, 0}] = 0.5f;
    layer_single.weights.value[{0, 1}] = -1.0f;
    layer_single.weights.value[{0, 2}] = 0.25f;
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model_single {loss_fn, opt};
    model_single.add_layer(layer_single);
    model_single.init();
    
    LinAlg::Tensor<float> X_single {{1, 2}};
    X_single[{0, 0}] = 1.5f;
    X_single[{0, 1}] = -2.0f;
    LinAlg::Tensor<float> Y_single {{1, 1}};
    Y_single[{0, 0}] = 0.75f;

    model_single.forward_pass(X_single);
    model_single.backpropagation(Y_single);
    model_single.optimizer_step();

    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer_batch {2, 1};
    layer_batch.weights.value[{0, 0}] = 0.5f;
    layer_batch.weights.value[{0, 1}] = -1.0f;
    layer_batch.weights.value[{0, 2}] = 0.25f;
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model_batch {loss_fn, opt};
    model_batch.add_layer(layer_batch);
    model_batch.init();

    LinAlg::Tensor<float> X_batch {{3, 2}};
    LinAlg::Tensor<float> Y_batch {{3, 1}};
    for(int b {}; b < 3; ++b) {
        X_batch[{b, 0}] = 1.5f;
        X_batch[{b, 1}] = -2.0f;
        Y_batch[{b, 0}] = 0.75f;
    }

    model_batch.forward_pass(X_batch);
    model_batch.backpropagation(Y_batch);
    model_batch.optimizer_step();

    EXPECT_TRUE(LinAlg::all_close<float>(model_batch.get_parameters()[0]->value, model_single.get_parameters()[0]->value, 1e-5f, 1e-5f));
}

TEST(Model, TestLoopMathcesCalculateLoss) {
    Func::MSE<float> loss_fn {};
    NN::Gradient_descent<float> opt {0.01f};
    NN::Layer<float, Func::Linear<float>, Func::No_activation<float>> layer {2, 1};
    layer.weights.value[{0, 0}] = 0.5f;
    layer.weights.value[{0, 1}] = -1.0f;
    layer.weights.value[{0, 2}] = 0.25f;
    NN::Model<float, Func::MSE<float>, NN::Gradient_descent<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2, 2}};
    X[{0, 0}] = 1.0f;
    X[{0, 1}] = 2.0f;
    X[{1, 0}] = -1.0f;
    X[{1, 1}] = 0.5f;

    LinAlg::Tensor<float> Y {{2, 1}};
    Y[{0, 0}] = 1.0f;
    Y[{1, 0}] = 0.0f;

    model.forward_pass(X);

    EXPECT_NEAR(model.calculate_loss(Y), model.test_loop(X, Y), 1e-5f);
}

TEST(Model, PredictionTargetBatchNotMatchingThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3};
    Func::MSE<float> loss_fn {};
    NN::Adam<float> opt {};
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> X {{2, 2}};
    model.forward_pass(X);

    LinAlg::Tensor<float> Y {{3, 3}};
    EXPECT_THROW(model.backpropagation(Y), std::invalid_argument);
}

TEST(Model, CalculateLossBeforeForwardPassThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3};
    Func::MSE<float> loss_fn {};
    NN::Adam<float> opt {};
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> Y {{1, 3}};
    EXPECT_THROW(model.calculate_loss(Y), std::invalid_argument);
}

TEST(Model, BackpropagationBeforeForwardPassThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3};
    Func::MSE<float> loss_fn {};
    NN::Adam<float> opt {};
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    LinAlg::Tensor<float> Y {{1, 3}};
    EXPECT_THROW(model.backpropagation(Y), std::invalid_argument);
}

TEST(Model, InitTwiceThrows) {
    NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> layer {2, 3};
    Func::MSE<float> loss_fn {};
    NN::Adam<float> opt {};
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};
    model.add_layer(layer);
    model.init();

    EXPECT_THROW(model.init(), std::invalid_argument);
}

TEST(Model, LoadingLayerIndexMismatchThrows) {
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {Func::MSE<float>{}, NN::Adam<float>{}};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>>{2, 4});
    model_load.init();

    std::filesystem::path path {"layer_index_mismatch.weights"};
    {
        std::ofstream out {path};
        out << "1\n5 4 3\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n";
    }

    EXPECT_THROW(model_load.load_weights(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(Model, LoadingWeightsPerNodeMismatchThrows) {
    std::filesystem::path dir {"saving_loading_tests"};
    std::filesystem::create_directories(dir);
    std::filesystem::path path {dir / "WeightsPerNodeMismatch.weights"};

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_save {Func::MSE<float>{}, NN::Adam<float>{}};
    model_save.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model_save.init();
    model_save.save_weights(path.string());

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {Func::MSE<float>{}, NN::Adam<float>{}};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {3, 3});
    model_load.init();

    EXPECT_THROW(model_load.load_weights(path.string()), std::invalid_argument);

    std::filesystem::remove_all(dir);
}

TEST(Model, SavingLoadingMultLyaerPredictionsMatch) {
    std::filesystem::path dir {"saving_loading_tests"};
    std::filesystem::create_directories(dir);
    std::filesystem::path path {dir / "MultiLayerRoundTrip.weights"};

    Rand::Random<float> random {};

    LinAlg::Tensor<float> samples {{20, 2}};
    samples.normal(random, 0, 1);

    NN::Adam<float> opt {};
    Func::MSE<float> loss_fn {};

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {loss_fn, opt};
    model.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model.add_layer(NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> {3, 2});
    model.init(random, samples);
    model.save_weights(path.string());

    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model_load {loss_fn, opt};
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::ReLU<float>> {2, 3});
    model_load.add_layer(NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> {3, 2});
    model_load.init(random, samples);
    model_load.load_weights(path.string());

    LinAlg::Tensor<float> X {{4, 2}};
    X.normal(random, 0, 1);

    EXPECT_EQ(model.forward_pass(X), model_load.forward_pass(X));

    std::filesystem::remove_all(dir);
}

TEST(Model, TrainLoopEpochsLessThanOneThrows) {
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {Func::MSE<float>{}, NN::Adam<float>{}};
    model.add_layer(NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> {2, 3});
    model.init();

    Rand::Random<float> random {42};

    LinAlg::Tensor<float> X {{20, 2}};
    LinAlg::Tensor<float> Y {{20, 3}};

    EXPECT_THROW(model.train_loop(random, X, Y, 0, 2), std::invalid_argument);
    EXPECT_THROW(model.train_loop(random, X, Y, -1, 2), std::invalid_argument);
}

TEST(Model, TrainLoopBatchSizeLessThanOneThrows) {
    NN::Model<float, Func::MSE<float>, NN::Adam<float>> model {Func::MSE<float>{}, NN::Adam<float>{}};
    model.add_layer(NN::Layer<float, Func::Linear<float>, Func::Sigmoid<float>> {2, 3});
    model.init();

    Rand::Random<float> random {42};

    LinAlg::Tensor<float> X {{20, 2}};
    LinAlg::Tensor<float> Y {{20, 3}};

    EXPECT_THROW(model.train_loop(random, X, Y, 4, 0), std::invalid_argument);
    EXPECT_THROW(model.train_loop(random, X, Y, 4, -1), std::invalid_argument);
}