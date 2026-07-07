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
        // Should zero-initilize all weight, T should be the input size, ie the nodes in previos layer
        std::constructible_from<F, T> &&
        requires(F f, const LinAlg::Tensor<T>& X, int i, 
                 Rand::Random<T>& random, T mean, T stddev,
                 T low, T high) {
            // Should set weights using normal distrobution
            {f.normal(random, mean, stddev)};
            // Should set weights using uniform distrobution
            {f.uniform(random, low, high)};
            // Takes in tensor of shape (...batch, 1, input layer size) and outputs result of function
            {f.function(X)} -> std::same_as<LinAlg::Tensor<T>>;
            // Takes in tensor of shape (...batch, 1, input layer size) and outputs tensor (1, nodes) the result of the derivate each input function
            {f.derivate(X)} -> std::same_as<LinAlg::Tensor<T>>;
            // Takes in tensor of shape (...batch, 1, input layer size) and returns the derivate of the function with respective to each weight in order of get_weight()
            {f.derivate_all_weights(X)} -> std::same_as<LinAlg::Tensor<T>>;
            // Returns the number of weights
            {f.num_weights()} -> std::same_as<T>;
        };

    template <typename F, typename T>
    concept Activation_function = 
        std::constructible_from<F> &&
        requires(F f, const LinAlg::Tensor<T>& X) {
            // Forward pass through the activation function
            // Input size (...batch, 1, layer size)
            // Output size (...batch, 1, layer size)
            {f.activate(X)} -> std::same_as<LinAlg::Tensor<T>>;
            // Derivate each function
            // Input size (...batch, 1, layer size)
            // Output size (...batch, 1 layer size)
            {f.derivate(X)} -> std::same_as<LinAlg::Tensor<T>>;
        };
}

#endif