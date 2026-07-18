#include <Func/std_func.h>
#include <LinAlg/tensor.h>
#include <gtest/gtest.h>

TEST(StdFunc, LinearNumWeights) {
    int size = Func::Linear<float>::num_weights(3);

    EXPECT_EQ(size, 4);
}

TEST(StdFunc, LinearMath) {
    LinAlg::Tensor<float> X {{1, 3}, 3};
    X[{0, 1}] = 8;
    X[{0, 2}] = -3;

    LinAlg::Tensor<float> weights {{1, 4}, 1};
    weights[{0, 0}] = 5;
    weights[{0, 2}] = -2;
    weights[{0, 3}] = -2;

    float func = Func::Linear<float>::function(X, weights);

    EXPECT_EQ(func, 27);

    LinAlg::Tensor<float> dF {Func::Linear<float>::function_grad(X, weights)};
    LinAlg::Tensor<float> dF_exp {{1, 3}};
    for(int i {}; i < 3; ++i) {
        dF_exp[{0, i}] = weights[{0, i}];
    }

    EXPECT_EQ(dF, dF_exp);

    LinAlg::Tensor<float> dW {Func::Linear<float>::weights_grad(X, weights)};
    LinAlg::Tensor<float> dW_exp {{1, 4}};
    for(int i {}; i < 3; ++i) {
        dW_exp[{0, i}] = X[{0, i}];
    }
    dW_exp[{0, 3}] = 1;

    EXPECT_EQ(dW, dW_exp);
}

TEST(StdFunc, ReLUMath) {
    LinAlg::Tensor<float> X {{1, 3}, 2};

    X[{0, 0}] = -1;
    X[{0, 2}] = 0;

    LinAlg::Tensor<float> A {Func::ReLU<float>::activate(X)};
    LinAlg::Tensor<float> A_exp {{1, 3}};
    A_exp[{0, 1}] = 2;

    EXPECT_EQ(A, A_exp);

    LinAlg::Tensor<float> D {Func::ReLU<float>::derivate(X)};
    LinAlg::Tensor<float> D_exp {{1, 3}};

    D_exp[{0, 1}] = 1;

    EXPECT_EQ(D, D_exp);
}

TEST(StdFunc, NoActivationMath) {
    LinAlg::Tensor<float> X {{1, 3}};
    X[{0, 0}] = 1;
    X[{0, 1}] = -7;

    LinAlg::Tensor<float> A {Func::No_Activation<float>::activate(X)};

    EXPECT_EQ(X, A);

    LinAlg::Tensor<float> D {Func::No_Activation<float>::derivate(X)};
    LinAlg::Tensor<float> D_exp {{1, 3}, 1};

    EXPECT_EQ(D, D_exp);
}