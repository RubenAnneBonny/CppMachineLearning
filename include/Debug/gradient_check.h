#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

#include <Func/function.h>
#include <LinAlg/tensor.h>
#include <cmath>

namespace Debug {
    namespace helper {
        template <std::floating_point T, 
                  std::invocable Eval>
        std::pair<T, int> max_rel_error(LinAlg::Tensor<T>& disturb,
                                        const LinAlg::Tensor<T>& analytic,
                                        Eval evaluate,
                                        T eps) {
            T max_rel_error {};
            int worst_index {-1};

            for(int i {}; i < disturb.get_extent(1); ++i) {
                T original {disturb[{0, i}]};

                disturb[{0, i}] = original + eps;
                T up {evaluate()};

                disturb[{0, i}] = original - eps;
                T down {evaluate()};

                disturb[{0, i}] = original;

                T numeric_val {(up - down) / (2 * eps)};
                T analytic_val {analytic[{0, i}]};

                T max = 1e-6;
                max = (max > std::abs(numeric_val) ? max : std::abs(numeric_val));
                max = (max > std::abs(analytic_val) ? max : std::abs(analytic_val));

                T rel_error {std::abs(numeric_val - analytic_val) / max};

                if(rel_error > max_rel_error) {
                    max_rel_error = rel_error;
                    worst_index = i;
                }
            }

            return {max_rel_error, worst_index};
        }
    }

    /// @brief A struct of valuable info about the result of gradient check for functions
    /// @tparam T A floating point type
    template <std::floating_point T>
    struct Func_grad_check_result {
        bool passed = false;
        T max_func_rel_error = 0;
        T max_weights_rel_error = 0;
        int worst_func_index = -1;
        int worst_weights_index = -1;
    };

    /// @brief Numerically tests a function to look for flaws in gradient calculation
    /// @tparam T A floating point type
    /// @tparam F The function to test
    /// @param input_size The size of input, don't need to be large if speed is necessary
    /// @param eps Step size disturbation, error in result is O(eps^2)
    /// @param rel_tol The relative tolerance, controls the passed variable in result
    /// @return A Func_grad_check_result instance
    template <std::floating_point T,
              Func::Function<T> F>
    Func_grad_check_result<T> func_grad_check(int input_size, T eps = 1e-5, T rel_tol = 1e-4) {
        LinAlg::Tensor<T> input {{1, input_size}};
        LinAlg::Tensor<T> weights {{1, F::num_weights(input_size)}};

        input.normal(0, 1);
        weights.normal(0, 1);

        Func_grad_check_result<T> result;

        auto eval {
            [&]()
            {
                return F::function(input, weights);
            }
        };

        LinAlg::Tensor<T> analytic_grad {F::function_grad(input, weights)};
        auto [func_rel_error, func_index] = helper::max_rel_error(input, analytic_grad, eval, eps);

        LinAlg::Tensor<T> analytic_weight_grad {F::weights_grad(input, weights)};
        auto [weight_rel_error, weight_index] = helper::max_rel_error(weights, analytic_weight_grad, eval, eps);

        result.max_func_rel_error = func_rel_error;
        result.worst_func_index = func_index;
        result.max_weights_rel_error = weight_rel_error;
        result.worst_weights_index = weight_index;
        result.passed = func_rel_error <= rel_tol && weight_rel_error <= rel_tol;

        return result;
    }

    /// @brief A struct of valuable info about the result of gradient chekc for activation functions
    /// @tparam T A floating point type
    template <std::floating_point T>
    struct Activation_grad_check_result {
        bool passed = false;
        T max_rel_error = 0;
        int worst_index = -1;
    };

    /// @brief Numerically tests a activation function to look for flaws in gradient calculation
    /// @tparam T A floating point type
    /// @tparam F The function to test
    /// @param input_size The size of input, don't need to be large if speed is necessary
    /// @param eps Step size disturbation, error in result is O(eps^2)
    /// @param rel_tol The relative tolerance, controls the passed variable in result
    /// @return A Activation_grad_check_result instance
    template <std::floating_point T,
              Func::Activation_function<T> A>
    Activation_grad_check_result<T> activation_grad_check(int input_size, T eps = 1e-5, T rel_tol = 1e-4) {
        LinAlg::Tensor<T> input {{1, input_size}};

        input.normal(0, 1);

        T margin {1e-2};
        for(int i {}; i < input_size; ++i) {
            if(std::abs(input[{0, i}]) < margin) {
                input[{0, i}] += (input[{0, i}] >= 0 ? margin : -margin);
            }
        }

        Activation_grad_check_result<T> result;

        auto eval {
            [&]()
            {
                return A::activate(input).sum();
            }
        };

        LinAlg::Tensor<T> analytic_grad {A::derivate(input)};
        auto [rel_error, index] = helper::max_rel_error(input, analytic_grad, eval, eps);

        result.max_rel_error = rel_error;
        result.worst_index = index;
        result.passed = rel_error <= rel_tol;

        return result;
    }
}

#endif