#ifndef FUNCTION_H
#define FUNCTION_H

#include <vector>
#include <Rand/random.h>
#include <stdexcept>
#include <concepts>
#include <LinAlg/tensor.h>

namespace Func{
    template <typename F, typename T>
    concept Function = 
        requires(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights, int i, int input_size) {
            /// @brief Returns the number of weights for the function, based on input size
            {F::num_weights(input_size)} -> std::same_as<int>;
            /// @brief Based on input and weights, calculates the output of the function
            /// @param X Input tensor of shape (1, input size)
            /// @param weights Tensor of shape (1, num_weights)
            /// @return Scalar
            {F::function(X, weights)} -> std::same_as<T>;
            /// @brief Calculates the gradient of the function, based on input
            /// @param X Input tensor of shape (1, input size)
            /// @param weights Tensor of shape (1, num_weights)
            /// @returns Tensor of shape (1, input_size)
            {F::function_grad(X, weights)} -> std::same_as<LinAlg::Tensor<T>>;
            /// @brief Calculates the gradient of the function, based on weights
            /// @param X Input tensor of shape (1, input size)
            /// @param weights Tensor of shape (1, num_weights)
            /// @return Tensor of shape (1, num_weights)
            {F::weights_grad(X, weights)} -> std::same_as<LinAlg::Tensor<T>>;
        };

    template <typename F, typename T>
    concept Activation_function = 
        requires(const LinAlg::Tensor<T>& X) {
            /// @brief Forward pass through the activation function
            /// @param X Tensor of shape (1, nodes)
            /// @return Tensor of shape (1, nodes)
            {F::activate(X)} -> std::same_as<LinAlg::Tensor<T>>;
            /// @brief Calculate gradient of Activation
            /// @param X Tensor of shape (1, nodes)
            /// @return Tensor of shape (1, nodes)
            {F::derivate(X)} -> std::same_as<LinAlg::Tensor<T>>;
        };

    template <typename F, typename T>
    concept Loss_function = 
        requires(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
            /// @brief Calculates the loss
            /// @param prediction The output from the neural network, tensor of shape (1, input size)
            /// @param target The target output, tensor of shape (1, input size)
            /// @return The loss, a scalar
            {F::loss(prediction, target)} -> std::same_as<T>;
            /// @brief Calculates the gradient of the loss function
            /// @param prediction The output from the neural network, tensor of shape (1, input size)
            /// @param target The target output, tensor of shape (1, input size)
            /// @result The gradient tensor of shape (1, input size)
            {F::gradient(prediction, target)} -> std::same_as<LinAlg::Tensor<T>>;
        };
}

#endif