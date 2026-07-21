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

TEST(Tensor, MultiplicationTensorScalar) {
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

TEST(Tensor, MultiplicationScalarTensor) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> C {(3 * A)};
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

TEST(Tensor, AdditionTensorScalar) {
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

TEST(Tensor, AdditionScalarTensor) {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> B {6 + A};
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

TEST(Tensor, RowView) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> A_1 {A.row(1)};

    A_1[{2}] = 6;
    A_1[{0}] = -4;

    LinAlg::Tensor<float> A_exp {{2, 3}, 1};
    A_exp[{0, 1}] = 3;
    A_exp[{1, 0}] = -4;
    A_exp[{1, 1}] = -2;
    A_exp[{1, 2}] = 6;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, CopyView) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> A_view {A};
    A_view[{0, 0}] = 3;
    A_view[{1, 1}] = 11;

    LinAlg::Tensor<float> A_exp {{2, 3}, 1};
    A_exp[{0, 0}] = 3;
    A_exp[{0, 1}] = 3;
    A_exp[{1, 1}] = 11;
    A_exp[{1, 2}] = -7;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, Sum) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    EXPECT_EQ(A.sum(), -3);
}

TEST(Tensor, Max) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 0}] = 3;
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    EXPECT_EQ(A.max(), 3);

    LinAlg::Tensor<float> B {{2, 2}, 3};
    B[{0, 0}] = -8;
    B[{0, 1}] = -11;
    B[{1, 0}] = -3,
    B[{1, 1}] = -4;

    EXPECT_EQ(B.max(), -3);
}

TEST(Tensor, Argmax) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    EXPECT_EQ(A.argmax(), std::vector<int>({0, 1}));

    LinAlg::Tensor<float> B {{2, 2}, -1};
    B[{0, 0}] = -8;
    B[{0, 1}] = -3;
    B[{1, 1}] = -27;

    EXPECT_EQ(B.argmax(), std::vector<int>({1, 0}));

    LinAlg::Tensor<float> C {{2, 2, 2}, 6};

    EXPECT_EQ(C.argmax(), std::vector<int>({0, 0, 0}));
}

TEST(Tensor, PairwiseSubtraction) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = -2;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {{3}, 2};
    B[{1}] = 5;

    LinAlg::Tensor<float> A_B {A - B};
    LinAlg::Tensor<float> A_B_exp {{2, 3}, -1};
    A_B_exp[{0, 1}] = -2;
    A_B_exp[{1, 1}] = -7;
    A_B_exp[{1, 2}] = -9;

    EXPECT_EQ(A_B, A_B_exp);

    LinAlg::Tensor<float> B_A {B - A};
    LinAlg::Tensor<float> B_A_exp {{2, 3}, 1};
    B_A_exp[{0, 1}] = 2;
    B_A_exp[{1, 1}] = 7;
    B_A_exp[{1, 2}] = 9;

    EXPECT_EQ(B_A, B_A_exp);
}

TEST(Tensor, SubtractionTensorScalar) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {A - 3};
    LinAlg::Tensor<float> B_exp {{2, 3}, -2};
    B_exp[{0, 1}] = 0;
    B_exp[{1, 1}] = 5;
    B_exp[{1, 2}] = -10; 

    EXPECT_EQ(B, B_exp);
}

TEST(Tensor, SubtractionScalarTensor) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {3 - A};
    LinAlg::Tensor<float> B_exp {{2, 3}, 2};
    B_exp[{0, 1}] = 0;
    B_exp[{1, 1}] = -5;
    B_exp[{1, 2}] = 10; 

    EXPECT_EQ(B, B_exp);
}

TEST(Tensor, Negation) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {-A};
    LinAlg::Tensor<float> B_exp {{2, 3}, -1};
    B_exp[{0, 1}] = -3;
    B_exp[{1, 1}] = -8;
    B_exp[{1, 2}] = 7;

    EXPECT_EQ(B, B_exp);
}

TEST(Tensor, SubtractionAssignmentTensor) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;
    
    LinAlg::Tensor<float> B {{2, 3}, 1};
    B[{0, 1}] = -1;
    B[{1, 2}] = 3;

    A -= B;

    LinAlg::Tensor<float> A_exp {{2, 3}, 0};
    A_exp[{0, 1}] = 4;
    A_exp[{1, 1}] = 7;
    A_exp[{1, 2}] = -10;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, SubtractionAssignmentScalar) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    A -= 7;

    LinAlg::Tensor<float> A_exp {{2, 3}, -6};
    A_exp[{0, 1}] = -4;
    A_exp[{1, 1}] = 1;
    A_exp[{1, 2}] = -14;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, AddAssignMutatesSharedStorage) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {{2, 3}, 1};
    B[{0, 1}] = -1;
    B[{1, 2}] = 3;

    LinAlg::Tensor<float> A_view {A};

    A += B;

    LinAlg::Tensor<float> A_view_exp {{2, 3}, 2};
    A_view_exp[{0, 1}] = 2;
    A_view_exp[{1, 1}] = 9;
    A_view_exp[{1, 2}] = -4;

    EXPECT_EQ(A_view, A_view_exp);
}

TEST(Tensor, SubAssignMutatesSharedStorage) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;
    A[{1, 2}] = -7;

    LinAlg::Tensor<float> B {{2, 3}, 1};
    B[{0, 1}] = -1;
    B[{1, 2}] = 3;

    LinAlg::Tensor<float> A_view {A};

    A -= B;

    LinAlg::Tensor<float> A_view_exp {{2, 3}, 0};
    A_view_exp[{0, 1}] = 4;
    A_view_exp[{1, 1}] = 7;
    A_view_exp[{1, 2}] = -10;

    EXPECT_EQ(A_view, A_view_exp);
}

TEST(Tensor, MatMulAssignMutatesSharedStorage) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 2}, 1};
    B[{0, 0}] = -1;
    B[{1, 1}] = 3;

    LinAlg::Tensor<float> A_view {A};

    A *= B;

    LinAlg::Tensor<float> A_view_exp {{2, 2}, 2};
    A_view_exp[{0, 1}] = 10;
    A_view_exp[{1, 0}] = 7;
    A_view_exp[{1, 1}] = 25;

    EXPECT_EQ(A_view, A_view_exp);
}

TEST(Tensor, MultAssignMutatesSharedStorage) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> A_view {A};

    A *= 3;

    LinAlg::Tensor<float> A_view_exp {{2, 2}, 3};
    A_view_exp[{0, 1}] = 9;
    A_view_exp[{1, 1}] = 24;

    EXPECT_EQ(A_view, A_view_exp);
}

TEST(Tensor, MatMulAssign) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 2}, 1};
    B[{0, 0}] = -1;
    B[{1, 1}] = 3;

    A *= B;

    LinAlg::Tensor<float> A_exp {{2, 2}, 2};
    A_exp[{0, 1}] = 10;
    A_exp[{1, 0}] = 7;
    A_exp[{1, 1}] = 25;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, MatMulAssignShapeChangeThrows) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 2, 2}, 1};
    B[{0, 0, 0}] = -1;
    B[{1, 1, 1}] = 3;

    EXPECT_THROW(A *= B, std::invalid_argument);
}

TEST(Tensor, AddAssignShapeChangeThrows) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 2, 2}, 1};
    B[{0, 0, 0}] = -1;
    B[{1, 1, 1}] = 3;

    EXPECT_THROW(A += B, std::invalid_argument);
}

TEST(Tensor, SubAssignShapeChangeThrows) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    A[{0, 1}] = 3;
    A[{1, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 2, 2}, 1};
    B[{0, 0, 0}] = -1;
    B[{1, 1, 1}] = 3;

    EXPECT_THROW(A -= B, std::invalid_argument);
}

TEST(Tensor, AllCloseWithinAtol) {
    LinAlg::Tensor<float> A {{2, 2}, 1.3f};

    LinAlg::Tensor<float> B {A + 1.003f};
    B -= 1.003f;

    EXPECT_TRUE(LinAlg::all_close<float>(A, B, 1e-6, 0));
}

TEST(Tensor, AllCloseOutsideTol) {
    LinAlg::Tensor<float> A {{2, 2}, 1.3f};

    LinAlg::Tensor<float> B {A + 1.003f};

    EXPECT_FALSE(LinAlg::all_close<float>(A, B, 1e-6, 0));
}

TEST(Tensor, AllCloseRtolScaling) {
    LinAlg::Tensor<float> A {{2, 2}, 1e7 + 1.3f};

    LinAlg::Tensor<float> B {A * 52.3f};

    B *= (1 / 52.3f);

    EXPECT_TRUE(LinAlg::all_close<float>(A, B, 0, 1e-6));
}

TEST(Tensor, AllCloseShapeMismatch) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    LinAlg::Tensor<float> B {{2, 2, 2}, 1};

    EXPECT_FALSE(LinAlg::all_close<float>(A, B, 1, 1));
}

TEST(Tensor, OneHot) {
    LinAlg::Tensor<float> A {LinAlg::one_hot<float>(4, 2)};
    LinAlg::Tensor<float> A_exp {{4}};
    A_exp[{2}] = 1;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, IndependentCopy) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A[{0, 2}] = 3;

    LinAlg::Tensor<float> B {A.copy()};

    A[{0, 1}] = 4;
    B[{0, 1}] = -1;

    EXPECT_NE(A, B);
}

TEST(Tensor, UnsqueezeIsView) {
    LinAlg::Tensor<float> A {{2}, 1};
    LinAlg::Tensor<float> A_unsqueezed {A.unsqueeze()};

    A_unsqueezed[{0, 1}] = 3;

    LinAlg::Tensor<float> A_exp {{2}, 1};
    A_exp[{1}] = 3;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, SqueezeIsView) {
    LinAlg::Tensor<float> A {{1, 2}, 1};
    LinAlg::Tensor<float> A_squeezed {A.squeeze()};

    A_squeezed[{0}] = -4;

    LinAlg::Tensor<float> A_exp {{1, 2}, 1};
    A_exp[{0, 0}] = -4;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, TransposeIsView) {
    LinAlg::Tensor<float> A {{2, 2}, 1};
    LinAlg::Tensor<float> A_t {A.t()};
    A_t[{0, 1}] = -4;

    LinAlg::Tensor<float> A_exp {{2, 2}, 1};
    A_exp[{1, 0}] = -4;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, RowThenUnsqueeze) {
    LinAlg::Tensor<float> A {{2, 2}};
    LinAlg::Tensor<float> A_ru {A.row(1).unsqueeze()};

    A_ru[{0, 1}] = 8;

    LinAlg::Tensor<float> A_exp {{2, 2}};
    A_exp[{1, 1}] = 8;

    EXPECT_EQ(A, A_exp);
}

TEST(Tensor, SumOfRowView) {
    LinAlg::Tensor<float> A {{2, 2}, 0};
    A[{0, 1}] = 8;
    A[{0, 0}] = -1;
    A[{1, 0}] = 27;

    EXPECT_EQ(7, A.row(0).sum());
}

TEST(Tensor, MaxOfRowView) {    
    LinAlg::Tensor<float> A {{2, 2}, 0};
    A[{0, 1}] = 8;
    A[{0, 0}] = -1;
    A[{1, 0}] = 27;

    EXPECT_EQ(8, A.row(0).max());
}

TEST(Tensor, MatmulTransposedView) {
    LinAlg::Tensor<float> A {{2, 2}, 0};
    A[{0, 0}] = 2;
    A[{0, 1}] = 4;
    A[{1, 1}] = -1;

    LinAlg::Tensor<float> B {{2, 2}, 1};
    B[{1, 0}] = 4;

    LinAlg::Tensor<float> C {A * B.t()};

    LinAlg::Tensor<float> C_exp {{2, 2}, 6};
    C_exp[{0, 1}] = 12;
    C_exp[{1, 0}] = -1;
    C_exp[{1, 1}] = -1;

    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, MatmulRankLessThanTwoThrows) {
    LinAlg::Tensor<float> A {{2}};
    LinAlg::Tensor<float> B {{2, 2}};

    EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(Tensor, MatmulInnerDimMismatchThrows) {
    LinAlg::Tensor<float> A {{2, 2}};
    LinAlg::Tensor<float> B {{3, 2}};

    EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(Tensor, MatmulBatchAxisMismatchThrows) {
    LinAlg::Tensor<float> A {{2, 2, 2}};
    LinAlg::Tensor<float> B {{3, 2, 2}};

    EXPECT_THROW(A * B, std::invalid_argument);
}

TEST(Tensor, GetExtentAxisOutOfRangeThrows) {
    LinAlg::Tensor<float> A {{2, 2}};

    EXPECT_THROW(A.get_extent(2), std::invalid_argument);
    EXPECT_THROW(A.get_extent(-1), std::invalid_argument);
}

TEST(Tensor, RowRankOneThrows) {
    LinAlg::Tensor<float> A {{3}};

    EXPECT_THROW(A.row(1), std::invalid_argument);
}

TEST(Tensor, RowIndexOutOfRangeThrows) {
    LinAlg::Tensor<float> A{{3, 2}};

    EXPECT_THROW(A.row(3), std::invalid_argument);
    EXPECT_THROW(A.row(-1), std::invalid_argument);
}

TEST(Tensor, UsqueezeAxisOutOfRangeThrows) {
    LinAlg::Tensor<float> A {{3, 2}};

    EXPECT_NO_THROW(A.unsqueeze(2));
    EXPECT_THROW(A.unsqueeze(-1), std::invalid_argument);
    EXPECT_THROW(A.unsqueeze(3), std::invalid_argument);
}

TEST(Tensor, SqueezeRankOneThrows) {
    LinAlg::Tensor<float> A {{2}};

    EXPECT_THROW(A.squeeze(), std::invalid_argument);
}

TEST(Tensor, SqueezeAxisOutOfRangeThrows) {
    LinAlg::Tensor<float> A {{1, 1}};

    EXPECT_THROW(A.squeeze(-1), std::invalid_argument);
    EXPECT_THROW(A.squeeze(2), std::invalid_argument);
}

TEST(Tensor, SqueezeExtentNotOneThrows) {
    LinAlg::Tensor<float> A {{2, 2}};

    EXPECT_THROW(A.squeeze(), std::invalid_argument);
}

TEST(Tensor, TransposeRankOneThrows) {
    LinAlg::Tensor<float> A {{2}};

    EXPECT_THROW(A.t(), std::invalid_argument);
}

TEST(Tensor, ConstructorInitValues) {
    LinAlg::Tensor<float> A {{3, 2}, 4};

    for(int i {}; i < 3; ++i) {
        for(int j {}; j < 2; ++j) {
            float element {A[{i, j}]};
            EXPECT_EQ(4, element);
        }
    }
}

TEST(Tensor, SetAllElements) {
    LinAlg::Tensor<float> A {{2, 3}, 1};
    A.set_all_elements(-10);

    for(int i {}; i < 2; ++i) {
        for(int j {}; j < 3; ++j) {
            float element {A[{i, j}]};
            EXPECT_EQ(-10, element);
        }
    }
}

TEST(Tensor, GetExtent) {
    LinAlg::Tensor<float> A {{2, 3}};

    EXPECT_EQ(A.get_extent(0), 2);
    EXPECT_EQ(A.get_extent(1), 3);
}

TEST(Tensor, NormalFillsAllElements) {
    LinAlg::Tensor<float> A {{2, 3}, 10};
    Rand::Random<float> random {42};

    A.normal(random, -3, 1);

    for(int i {}; i < 2; ++i) {
        for(int j {}; j < 3; ++j) {
            float element {A[{i, j}]};
            EXPECT_NE(element, 10);
        }
    }

    float first {A[{0, 0}]};
    float second {A[{1, 2}]};

    EXPECT_NE(first, second);
}

TEST(Tensor, UniformWithinBounds) {
    LinAlg::Tensor<float> A {{2, 2}, 10};
    Rand::Random<float> random {42};

    A.uniform(random, 0, 3);

    for(int i {}; i < 2; ++i) {
        for(int j {}; j < 2; ++j) {
            float element {A[{i, j}]};
            EXPECT_TRUE(element >= 0 && element <= 3);
        }
    }
}

TEST(Tensor, ToString) {
    LinAlg::Tensor<float> A {{2, 3}, -3};
    A[{0, 0}] = 8;

    std::string A_str {static_cast<std::string>(A)};

    EXPECT_EQ(A_str, "(2, 3)");
}

TEST(Tensor, PairwiseAddFreeFunction) {
    LinAlg::Tensor<float> A {{2, 3}, -1};
    A[{0, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 3}, 5};
    B[{1, 2}] = 2;
    B[{0, 1}] = 1;

    LinAlg::Tensor<float> C {LinAlg::pairwise_add<float>(A, B)};
    LinAlg::Tensor<float> C_exp {{2, 3}, 4};
    C_exp[{0, 1}] = 9;
    C_exp[{1, 2}] = 1;
    
    EXPECT_EQ(C, C_exp);
}

TEST(Tensor, PairwiseSubFreeFunction) {
    LinAlg::Tensor<float> A {{2, 3}, -1};
    A[{0, 1}] = 8;

    LinAlg::Tensor<float> B {{2, 3}, 5};
    B[{1, 2}] = 2;
    B[{0, 1}] = 1;

    LinAlg::Tensor<float> C {LinAlg::pairwise_sub<float>(A, B)};
    LinAlg::Tensor<float> C_exp {{2, 3}, -6};
    C_exp[{0, 1}] = 7;
    C_exp[{1, 2}] = -3;
    
    EXPECT_EQ(C, C_exp);    
}

TEST(Tensor, SliceInvalidRangeThrows) {
    LinAlg::Tensor<float> A {{5, 2}, 2};

    EXPECT_THROW(A.slice(2, 2), std::invalid_argument);
    EXPECT_THROW(A.slice(-1, 3), std::invalid_argument);
    EXPECT_THROW(A.slice(3, 6), std::invalid_argument);
    EXPECT_THROW(A.slice(3, 2), std::invalid_argument);
}

TEST(Tensor, SliceIsView) {
    LinAlg::Tensor<float> A {{5, 2}, 2};
    LinAlg::Tensor<float> slice {A.slice(1, 2)};
    slice[{0, 1}] = 8;

    LinAlg::Tensor<float> A_exp {{5, 2}, 2};
    A_exp[{1, 1}] = 8;

    EXPECT_EQ(A, A_exp);
}