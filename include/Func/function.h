#ifndef FUNCTION_H
#define FUNCTION_H

/**
 * @file
 * @brief Defines the concepts for functions, activation functions and loss
 * functions
 * 
 * @code
 * template <std::floating_point T>
 * class My_activation {
 *      public:
 *          static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
 *              return X * 3;
 *          }
 *  
 *          static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
 *              return LinAlg::Tensor<T> {X.get_shape(), 3};
 *          }
 * };
 * @endcode
 */

#include <vector>
#include <Rand/random.h>
#include <stdexcept>
#include <concepts>
#include <LinAlg/tensor.h>

namespace Func{
    /**
     * @brief A concept for the normal function inside a node
     *
     * @details For all functions the input tensor X is shape (1, input_size) and
     * weight shape is (1, num_weights). num_weights should return the number of
     * weights the function uses based on the input_size. function should based
     * on input and weights, calulate the output of the function as a scalar.
     * The input_derivative should output a tensor of shape (1, input_size), the
     * gradient of the function with respective to each of its inputs.
     * weight_derivative should output tensor of shape (1, num_weights) the gradient
     * of the function with respective to each weight.
     *
     * @warning Make sure the weights have an internal ordering, for example the
     * second weight in the parameter weights, should correspond to the gradient
     * of the second weight when using weights_grad
     */
    template <typename F, typename T>
    concept Function = 
        requires(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights, int i, int input_size) {
            {F::num_weights(input_size)} -> std::same_as<int>;
            {F::function(X, weights)} -> std::same_as<T>;
            {F::input_derivative(X, weights)} -> std::same_as<LinAlg::Tensor<T>>;
            {F::weight_derivative(X, weights)} -> std::same_as<LinAlg::Tensor<T>>;
        };

    /**
     * @brief A concept for the activation function used in a layer
     *
     * @details For all functions the shape of the input tensor X is shape
     * (1, nodes). activate should perform the activation function on all
     * elements of a copy of the input tensor, the output should be of shape
     * (1, nodes). derivate should calculate the gradient of the activation
     * function and output a tensor of shape (1, nodes), where each (sample,
     * node) is the gradient of the activation function with respect to that
     * input.
     */
    template <typename F, typename T>
    concept Activation_function = 
        requires(const LinAlg::Tensor<T>& X) {
            {F::activate(X)} -> std::same_as<LinAlg::Tensor<T>>;
            {F::derivative(X)} -> std::same_as<LinAlg::Tensor<T>>;
        };

    /**
     * @brief A concept for the loss function used in a neural network
     *
     * @details For all functions the prediction and target tensors should be of
     * shape (1, input_size). The loss function should calculate the loss
     * value, a scalar. The gradient function should calculate a gradient tensor
     * of shape (1, input_size) the gradient of the function with respective
     * to each input.
     */
    template <typename F, typename T>
    concept Loss_function = 
        requires(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
            {F::loss(prediction, target)} -> std::same_as<T>;
            {F::gradient(prediction, target)} -> std::same_as<LinAlg::Tensor<T>>;
        };
}

#endif