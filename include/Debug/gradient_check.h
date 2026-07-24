#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

/*
    Gradient checking uses double, instantiate you're function/activation/loss with double
*/

#include <Func/function.h>
#include <LinAlg/tensor.h>
#include <Rand/random.h>
#include <cmath>

namespace Debug::GradCheck {
    namespace Helper {
        template <std::invocable Eval>
        std::pair<double, int> max_rel_error(LinAlg::Tensor<double>& disturb, const LinAlg::Tensor<double>& analytic, Eval evaluate, double eps) {
            double worst_rel_error {};
            int worst_index {-1};

            for(int i {}; i < disturb.get_extent(1); ++i) {
                double original {disturb[{0, i}]};

                disturb[{0, i}] = original + eps;
                double up {evaluate()};

                disturb[{0, i}] = original - eps;
                double down {evaluate()};

                disturb[{0, i}] = original;

                double numeric_val {(up - down) / (2 * eps)};
                double analytic_val {analytic[{0, i}]};

                double max = 1e-6;
                max = (max > std::abs(numeric_val) ? max : std::abs(numeric_val));
                max = (max > std::abs(analytic_val) ? max : std::abs(analytic_val));

                double rel_error {std::abs(numeric_val - analytic_val) / max};

                if(rel_error > worst_rel_error) {
                    worst_rel_error = rel_error;
                    worst_index = i;
                }
            }

            return {worst_rel_error, worst_index};
        }
    }

    /// @brief A struct of valuable info about the result of gradient check for functions
    struct Func_result {
        bool passed = false;
        double max_func_rel_error = 0;
        double max_weights_rel_error = 0;
        int worst_func_index = -1;
        int worst_weights_index = -1;
    };

    /// @brief Numerically tests a function to look for flaws in gradient calculation
    /// @tparam F The function to test
    /// @param input_size The size of input, don't need to be large if speed is necessary
    /// @param seed The seed, to ensure reproducability
    /// @param eps Step size disturbation, error in result is O(eps^2)
    /// @param rel_tol The relative tolerance, controls the passed variable in result
    /// @return A Func_result instance
    template <Func::Function<double> F>
    Func_result func(int input_size, int seed = 42, double eps = 1e-5, double rel_tol = 1e-6) {
        LinAlg::Tensor<double> input {{1, input_size}};
        LinAlg::Tensor<double> weights {{1, F::num_weights(input_size)}};

        Rand::Random<double> random {seed};

        input.normal(random, 0, 1);
        weights.normal(random, 0, 1);

        Func_result result;

        auto eval {
            [&]()
            {
                return F::function(input, weights);
            }
        };

        LinAlg::Tensor<double> analytic_grad {F::function_grad(input, weights)};
        auto [func_rel_error, func_index] = Helper::max_rel_error(input, analytic_grad, eval, eps);

        LinAlg::Tensor<double> analytic_weight_grad {F::weights_grad(input, weights)};
        auto [weight_rel_error, weight_index] = Helper::max_rel_error(weights, analytic_weight_grad, eval, eps);

        result.max_func_rel_error = func_rel_error;
        result.worst_func_index = func_index;
        result.max_weights_rel_error = weight_rel_error;
        result.worst_weights_index = weight_index;
        result.passed = func_rel_error <= rel_tol && weight_rel_error <= rel_tol;

        return result;
    }

    /// @brief A struct of valuable info about the result of gradient chekc for activation functions
    struct Activation_result {
        bool passed = false;
        double max_rel_error = 0;
        int worst_index = -1;
    };

    /// @brief Numerically tests a activation function to look for flaws in gradient calculation
    /// @tparam A The activation function to test
    /// @param input_size The size of input, don't need to be large if speed is necessary
    /// @param seed The seed to ensure reproducability
    /// @param eps Step size disturbation, error in result is O(eps^2)
    /// @param rel_tol The relative tolerance, controls the passed variable in result
    /// @return A Activation_result instance
    template <Func::Activation_function<double> A>
    Activation_result activation(int input_size, int seed = 42, double eps = 1e-5, double rel_tol = 1e-6) {
        LinAlg::Tensor<double> input {{1, input_size}};

        Rand::Random<double> random {seed};

        input.normal(random, 0, 1);

        double margin {1e-2};
        for(int i {}; i < input_size; ++i) {
            if(std::abs(input[{0, i}]) < margin) {
                input[{0, i}] += (input[{0, i}] >= 0 ? margin : -margin);
            }
        }

        Activation_result result;

        auto eval {
            [&]()
            {
                return A::activate(input).sum();
            }
        };

        LinAlg::Tensor<double> analytic_grad {A::derivate(input)};
        auto [rel_error, index] = Helper::max_rel_error(input, analytic_grad, eval, eps);

        result.max_rel_error = rel_error;
        result.worst_index = index;
        result.passed = rel_error <= rel_tol;

        return result;
    }

    /// @brief A struct of valuable info about the result of gradient check for loss functions
    struct Loss_result {
        bool passed = false;
        double max_rel_error = 0;
        int worst_index = -1;
    };

    /// @brief Numerically tests a loss function to look for flaws in gradient calculation
    /// @tparam L The loss function to test
    /// @param sample_target A target tensor of shape (1, x)
    /// @param seed The seed to ensure reproducability
    /// @param eps Step size disturbation, error in result is O(eps^2)
    /// @param rel_tol The relative tolerance, controls the passed variable in result
    /// @return A Loss_result instance
    /// @throws std::invalid_argument if sample_target rank isn't 2
    /// @throws std::invalid_argument if extent of first axis of sample_target isn't 1
    template <Func::Loss_function<double> L>
    Loss_result loss(const LinAlg::Tensor<double>& sample_target, int seed = 41251, double eps = 1e-5, double rel_tol = 1e-6) {
        if(sample_target.get_rank() != 2) {
            throw std::invalid_argument(
                "Cannot use target of shape " + 
                static_cast<std::string>(sample_target) + 
                " in gradient check, it must have rank 2"
            );
        }

        if(sample_target.get_extent(0) != 1) {
            throw std::invalid_argument(
                "Cannot use target of shape " + 
                static_cast<std::string>(sample_target) + 
                " in gradient check, the extent of the first axis must be 1"
            );
        }
        
        int input_size {sample_target.get_extent(1)};
        LinAlg::Tensor<double> prediction {{1, input_size}};
        Rand::Random<double> random {seed};

        prediction.normal(random, 0, 1);

        Loss_result result;

        auto eval {
            [&]()
            {
                return L::loss(prediction, sample_target);
            }
        };

        LinAlg::Tensor<double> analytic_grad {L::gradient(prediction, sample_target)};
        auto [rel_error, index] = Helper::max_rel_error(prediction, analytic_grad, eval, eps);

        result.max_rel_error = rel_error;
        result.worst_index = index;
        result.passed = rel_error <= rel_tol;

        return result;
    }
}

#endif