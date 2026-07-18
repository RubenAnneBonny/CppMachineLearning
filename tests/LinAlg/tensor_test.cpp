#include <LinAlg/tensor.h>
#include <gtest/gtest.h>

TEST(Tensor, AccessOperator) {
    LinAlg::Tensor<float> A {{2, 2}};

    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 3; A[{1, 1}] = 5;

    float sum = 0;
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < 2; ++j){
            sum += A[{i, j}];
        }
    }

    EXPECT_EQ(sum, 11);
}

TEST(Tensor, EqualityOperator) {
    LinAlg::Tensor<float> A {{2, 3}};
    LinAlg::Tensor<float> B {{2, 3}};
    LinAlg::Tensor<float> C {{2, 3}};
    LinAlg::Tensor<float> D {{1, 2, 3}};
    LinAlg::Tensor<float> E {{3, 3}};

    A[{1, 1}] = 1;
    B[{1, 1}] = 1;
    C[{0, 1}] = 1;
    D[{0, 1, 1}] = 1;
    E[{1, 1}] = 1;

    EXPECT_EQ(A, B);
    EXPECT_NE(A, C);
    EXPECT_NE(A, D);
    EXPECT_NE(A, E);
}

TEST(Tensor, MatrixMultiplication) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    LinAlg::Tensor<float> B {{3, 1}, 1};

    A[{0, 1}] = 2;
    A[{1, 2}] = 3;

    B[{1, 0}] = -4;
    B[{2, 0}] = 2;

    LinAlg::Tensor<float> C {A * B};
    LinAlg::Tensor<float> C_exp {{2, 1}, -5};

    C_exp[{1, 0}] = 3;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, Batching) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};
    LinAlg::Tensor<float> B {{3, 1}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    B[{1, 0}] = -4;
    B[{2, 0}] = 2;

    LinAlg::Tensor<float> C {A * B};
    LinAlg::Tensor<float> C_exp {{2, 2, 1}};

    C_exp[{0, 0, 0}] = -5;
    C_exp[{0, 1, 0}] = 3;
    C_exp[{1, 0, 0}] = -1;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, PairwiseMultiplication) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};
    LinAlg::Tensor<float> B {{2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    B[{1, 0}] = -4;
    B[{1, 1}] = 2;

    LinAlg::Tensor<float> C {LinAlg::pairwise_mult(A, B)};
    LinAlg::Tensor<float> C_exp {{2, 2, 3}, 1};

    C_exp[{0, 0, 1}] = 2;
    C_exp[{0, 1, 0}] = -4;
    C_exp[{0, 1, 1}] = 2;
    C_exp[{0, 1, 2}] = 3;
    C_exp[{1, 1, 0}] = -8;
    C_exp[{1, 1, 1}] = 2;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, ElementwiseMultiplication) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> C {(A * 3)};
    LinAlg::Tensor<float> C_exp {{2, 2, 3}, 3};

    C_exp[{0, 0, 1}] = 6;
    C_exp[{0, 1, 2}] = 9;
    C_exp[{1, 1, 0}] = 6;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, PairwiseAddition) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};
    LinAlg::Tensor<float> B {{2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    B[{1, 0}] = -4;
    B[{1, 1}] = 2;

    LinAlg::Tensor<float> C{A + B};
    LinAlg::Tensor<float> C_exp{{2, 2, 3}, 2};

    C_exp[{0, 0, 1}] = 3;
    C_exp[{0, 1, 0}] = -3;
    C_exp[{0, 1, 1}] = 3;
    C_exp[{0, 1, 2}] = 4;
    C_exp[{1, 1, 0}] = -2;
    C_exp[{1, 1, 1}] = 3;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, ElementwiseAddition) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> B {A + 6};
    LinAlg::Tensor<float> B_exp {{2, 2, 3}, 7};

    B_exp[{0, 0, 1}] = 8;
    B_exp[{0, 1, 2}] = 9;
    B_exp[{1, 1, 0}] = 8;

    EXPECT_EQ(B, B_exp);
}

TEST(Tensor, Unsqueeze) {
    LinAlg::Tensor<float> A {{2, 2}, 1};

    A[{1, 1}] = 3;

    A = A.unsqueeze(1);

    LinAlg::Tensor<float> A_exp {{2, 1, 2}, 1};

    A_exp[{1, 0, 1}] = 3;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, Squeeze) {
    LinAlg::Tensor<float> A {{2, 1, 2}, 1};

    A[{0, 0, 0}] = -5;

    A = A.squeeze(1);

    LinAlg::Tensor<float> A_exp ({2, 2}, 1);
    
    A_exp[{0, 0}] = -5;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, Transpose) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = -2;

    LinAlg::Tensor<float> B {A.t()};
    LinAlg::Tensor<float> B_exp {{2, 3, 2}, 1};

    B_exp[{0, 1, 0}] = 2;
    B_exp[{0, 2, 1}] = 3;
    B_exp[{1, 0, 1}] = -2;

    EXPECT_EQ(B, B_exp);
}

TEST(Tensor, Row) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = -2;

    LinAlg::Tensor<float> B {A.row(1)};
    LinAlg::Tensor<float> B_exp {{2, 3}, 1};

    B_exp[{1, 0}] = -2;

    EXPECT_EQ(B, B_exp);
}