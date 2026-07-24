#include <Func/function.h>
#include <LinAlg/tensor.h>
#include <Debug/gradient_check.h>
#include <gtest/gtest.h>

namespace {
    template <std::floating_point T>
    class Correct_func {
        public:
            static int num_weights(int input_size) {
                return input_size;
            }

            static T function(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                LinAlg::Tensor<T> cube {X * X * X};

                return (cube * weights).sum();
            }

            static LinAlg::Tensor<T> function_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return 3 * X * X * weights;
            }

            static LinAlg::Tensor<T> weights_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return X * X * X;
            }
    };

    template <std::floating_point T>
    class Wrong_func {
        public:
            static int num_weights(int input_size) {
                return Correct_func<T>::num_weights(input_size);
            }

            static T function(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::function(X, weights);
            }

            static LinAlg::Tensor<T> function_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::function_grad(X, weights) * 1.5;
            }

            static LinAlg::Tensor<T> weights_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::weights_grad(X, weights);
            }
    };

    template <std::floating_point T>
    class Wrong_weights {
        public:
            static int num_weights(int input_size) {
                return Correct_func<T>::num_weights(input_size);
            }

            static T function(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::function(X, weights);
            }

            static LinAlg::Tensor<T> function_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::function_grad(X, weights);
            }

            static LinAlg::Tensor<T> weights_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                return Correct_func<T>::weights_grad(X, weights) * 1.5;
            }
    };

    template <std::floating_point T>
    class Correct_activation {
        public:
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                return X * X;
            }

            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                return 2 * X;
            }
    };

    template <std::floating_point T>
    class Wrong_activation {
        public:     
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                return Correct_activation<T>::activate(X);
            }

            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                return Correct_activation<T>::derivate(X) * 1.5;
            }
    };

    template <std::floating_point T>
    class Correct_loss {
        public:
            static T loss(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                LinAlg::Tensor<T> diff {prediction - target};

                return (diff * diff).sum();
            }

            static LinAlg::Tensor<T> gradient(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                return 2 * (prediction - target);
            }
    };

    template <std::floating_point T>
    class Wrong_loss {
        public:
            static T loss(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                return Correct_loss<T>::loss(prediction, target);
            }

            static LinAlg::Tensor<T> gradient(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                return Correct_loss<T>::gradient(prediction, target) * 1.5;
            }
    };
}

TEST(GradientCheckTest, CorrectFunc) {
    Debug::GradCheck::Func_result result = Debug::GradCheck::func<Correct_func<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheckTest, WrongFunc) {
    Debug::GradCheck::Func_result result = Debug::GradCheck::func<Wrong_func<double>>(4);

    EXPECT_FALSE(result.passed);
}

TEST(GradientCheckTest, WrongWeights) {
    Debug::GradCheck::Func_result result = Debug::GradCheck::func<Wrong_weights<double>>(4);

    EXPECT_FALSE(result.passed);
}

TEST(GradientCheckTest, CorrectActivation) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Correct_activation<double>>(4);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheckTest, WrongActivation) {
    Debug::GradCheck::Activation_result result = Debug::GradCheck::activation<Wrong_activation<double>>(4);

    EXPECT_FALSE(result.passed);
}

TEST(GradientCheckTest, CorrectLoss) {
    Rand::Random<double> random {};
    LinAlg::Tensor<double> target {{1, 4}};
    target.normal(random, 0, 1);
    Debug::GradCheck::Loss_result result = Debug::GradCheck::loss<Correct_loss<double>>(target);

    EXPECT_TRUE(result.passed);
}

TEST(GradientCheckTest, WrongLoss) {
    Rand::Random<double> random {};
    LinAlg::Tensor<double> target {{1, 4}};
    target.normal(random, 0, 1);
    Debug::GradCheck::Loss_result result = Debug::GradCheck::loss<Wrong_loss<double>>(target);

    EXPECT_FALSE(result.passed);
}