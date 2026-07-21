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
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Func::No_Activation<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, Sigmoid) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Func::Sigmoid<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheck, MSE) {
    Debug::GradCheck::Loss_result result = Debug::GradCheck::loss<Func::MSE<double>>(4);

    EXPECT_TRUE(result.passed);
}