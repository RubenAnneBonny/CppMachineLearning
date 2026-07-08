#include <LinAlg/tensor.h>
#include <cassert>

void access_operator_test(){
    LinAlg::Tensor<float> A {{2, 2}};

    A[{0, 0}] = 1; A[{0, 1}] = 2;
    A[{1, 0}] = 3; A[{1, 1}] = 5;

    float sum = 0;
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < 2; ++j){
            sum += A[{i, j}];
        }
    }

    assert(sum == 11 && "Indexing in Tensor is incorrect");
}

void equality_operator_test(){
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

    assert(A == B && "Equality operator returns false between two identical tensors");
    assert(A != C && "Inequality operator returns false between two different tensors");
    assert(A != D && "Inequality operator returns false between two different tensors");
    assert(A != E && "Inequality operator returns false between two different tensors");
}

void matrix_multiplication_test(){
    LinAlg::Tensor<float> A {{2, 3}, 1};
    LinAlg::Tensor<float> B {{3, 1}, 1};

    A[{0, 1}] = 2;
    A[{1, 2}] = 3;

    B[{1, 0}] = -4;
    B[{2, 0}] = 2;

    LinAlg::Tensor<float> C {A * B};
    LinAlg::Tensor<float> C_exp {{2, 1}, -5};

    C_exp[{1, 0}] = 3;

    assert(C == C_exp && "Tensor multiplication between two matricies fails");
}

void batching_test(){
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

    assert(C == C_exp && "Batching in tensor multiplication does not work");
}

void pairwise_multiplication_test(){
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

    assert(C == C_exp && "Pairwise multiplication does not work");
}

void elementwise_multiplication_test(){
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> C {(A * 3)};
    LinAlg::Tensor<float> C_exp {{2, 2, 3}, 3};

    C_exp[{0, 0, 1}] = 6;
    C_exp[{0, 1, 2}] = 9;
    C_exp[{1, 1, 0}] = 6;

    assert(C == C_exp && "Elementwise multiplication does not work");
}

void pairwise_addition_test() {
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

    assert(C == C_exp && "Elementwise addition between tensors does not work");
}

void elementwise_addition_test() {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = 2;

    LinAlg::Tensor<float> B {A + 6};
    LinAlg::Tensor<float> B_exp {{2, 2, 3}, 7};

    B_exp[{0, 0, 1}] = 8;
    B_exp[{0, 1, 2}] = 9;
    B_exp[{1, 1, 0}] = 8;

    assert(B == B_exp && "Elementwise addition between tensor and float fails");
}

void unsqueeze_test() {
    LinAlg::Tensor<float> A {{2, 2}, 1};

    A[{1, 1}] = 3;

    A = A.unsqueeze(1);

    LinAlg::Tensor<float> A_exp {{2, 1, 2}, 1};

    A_exp[{1, 0, 1}] = 3;

    assert(A == A_exp && "Unsqueezing does not work");
}

void squeeze_test() {
    LinAlg::Tensor<float> A {{2, 1, 2}, 1};

    A[{0, 0, 0}] = -5;

    A = A.squeeze(1);

    LinAlg::Tensor<float> A_exp ({2, 2}, 1);
    
    A_exp[{0, 0}] = -5;

    assert(A == A_exp && "Squeezing does not work");
}

void t_test() {
    LinAlg::Tensor<float> A {{2, 2, 3}, 1};

    A[{0, 0, 1}] = 2;
    A[{0, 1, 2}] = 3;
    A[{1, 1, 0}] = -2;

    LinAlg::Tensor<float> B {A.t()};
    LinAlg::Tensor<float> B_exp {{2, 3, 2}, 1};

    B_exp[{0, 1, 0}] = 2;
    B_exp[{0, 2, 1}] = 3;
    B_exp[{1, 0, 1}] = -2;

    assert(B == B_exp && "Transposing does not work");
}

int main(){
    access_operator_test();
    equality_operator_test();
    matrix_multiplication_test();
    batching_test();
    pairwise_multiplication_test();
    elementwise_multiplication_test();
    pairwise_addition_test();
    elementwise_addition_test();
    unsqueeze_test();
    squeeze_test();
    t_test();

    return 0;
}