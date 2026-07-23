#include <Func/std_func.h>
#include <Debug/gradient_check.h>
#include <gtest/gtest.h>

TEST(GradientCheck, Linear) {
    Debug::GradCheck::Func_result result = Debug::GradCheck::func<Func::Linear<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, ReLU) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Func::ReLU<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, NoActivation) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Func::No_activation<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, Sigmoid) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Func::Sigmoid<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, MSE) {
    Rand::Random<double> random {42};
    LinAlg::Tensor<double> target {{1, 4}};
    target.normal(random, 0, 1);
    Debug::GradCheck::Loss_result result = Debug::GradCheck::loss<Func::MSE<double>>(target);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, Softmax_cross_entropy) {
    LinAlg::Tensor<double> target {LinAlg::one_hot<double>(4, 2).unsqueeze()};
    Debug::GradCheck::Loss_result result = Debug::GradCheck::loss<Func::Softmax_cross_entropy<double>>(target);

    EXPECT_TRUE(result.passed);
}