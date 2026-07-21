#include <NN/parameter.h>
#include <NN/std_opt.h>
#include <LinAlg/tensor.h>
#include <vector>
#include <gtest/gtest.h>

TEST(Optimizer, GDStep) {
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

TEST(Optimizer, GDMultipleParameters) {
    NN::Parameter<float> param_1 {{1, 3}, 1};
    param_1.value[{0, 1}] = 5;
    param_1.value[{0, 2}] = -2;

    param_1.grad[{0, 0}] = 8;
    param_1.grad[{0, 2}] = -5;

    NN::Parameter<float> param_2 {{1, 2}, 1};
    param_2.value[{0, 0}] = 3;
    param_2.grad[{0, 1}] = -2;

    NN::Gradient_descent<float> gd {2};

    std::vector<NN::Parameter<float>*> params;
    params.push_back(&param_1);
    params.push_back(&param_2);

    gd.init(params);
    gd.step();

    LinAlg::Tensor<float> p1_exp {{1, 3}, -15};
    p1_exp[{0, 1}] = 5;
    p1_exp[{0, 2}] = 8;

    LinAlg::Tensor<float> p2_exp {{1, 2}, 3};
    p2_exp[{0, 1}] = 5;

    EXPECT_EQ(param_1.value, p1_exp);
    EXPECT_EQ(param_2.value, p2_exp);
}