#include <NN/parameter.h>
#include <NN/std_opt.h>
#include <LinAlg/tensor.h>
#include <Debug/optimizer_check.h>
#include <vector>
#include <gtest/gtest.h>

TEST(Optimizer, GDOptimizerCheck) {
    NN::Gradient_descent<double> gd {0.01};

    Debug::Optimizer_result result {Debug::optimizer_check<NN::Gradient_descent<double>>(gd)};

    EXPECT_TRUE(result.passed);
}

TEST(Optimizer, GDHandcomputedStep) {
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

    EXPECT_EQ(param.value, val_exp);
}

TEST(Optimizer, AdamOptimizerCheck) {
    NN::Adam<double> adam {};

    Debug::Optimizer_result result {Debug::optimizer_check<NN::Adam<double>>(adam)};

    EXPECT_TRUE(result.passed);
}

TEST(Optimizer, AdamHandcomputedStep) {
    NN::Parameter<float> param {{1, 3}, 1};
    param.value[{0, 1}] = 5;
    param.value[{0, 2}] = -2;

    param.grad[{0, 0}] = 8;
    param.grad[{0, 2}] = -5;

    NN::Adam<float> adam {0.9f, 0.999f, 0.5f, 1e-8f};
    
    std::vector<NN::Parameter<float>*> params;
    params.push_back(&param);

    adam.init(params);
    adam.step();

    LinAlg::Tensor<float> value_exp {{1, 3}, 0.5f};
    value_exp[{0, 1}] = 5;
    value_exp[{0, 2}] = -1.5f;

    EXPECT_TRUE(LinAlg::all_close<float>(param.value, value_exp, 1e-5f, 1e-5f));
}

TEST(Optimizer, AdamMultipleSteps) {
    NN::Parameter<float> param {{1, 3}, 1};
    param.value[{0, 1}] = 5;
    param.value[{0, 2}] = -2;

    param.grad[{0, 0}] = 8;
    param.grad[{0, 2}] = -5;

    NN::Adam<float> adam {0.9f, 0.999f, 0.5f, 1e-8f};
    
    std::vector<NN::Parameter<float>*> params;
    params.push_back(&param);

    adam.init(params);
    adam.step();
    adam.step();
    adam.step();

    LinAlg::Tensor<float> value_exp {{1, 3}, -0.5f};
    value_exp[{0, 1}] = 5;
    value_exp[{0, 2}] = -0.5f;

    EXPECT_TRUE(LinAlg::all_close<float>(param.value, value_exp, 1e-5f, 1e-5f));
}