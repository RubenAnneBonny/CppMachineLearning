#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <stdexcept>
#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Func/function.h>

namespace NN {
    template <std::floating_point T>
    class Parameter {
        public:
            LinAlg::Tensor<T> value;
            LinAlg::Tensor<T> grad;

            Parameter(const std::vector<int>& shape, T init = 0) 
                : value {shape, init}
                , grad {shape}
            {}

            void uniform(const Rand::Random<T>& random, T low, T high) {
                value.uniform(random, low, high);
            }

            void normal(const Rand::Random<T>& random, T mean, T stddev) {
                value.normal(random, mean, stddev);
            }
    };
    
    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    class Layer {
        private:
            int m_nodes;
            int m_input_nodes;
            LinAlg::Tensor<T> m_store_X;
            LinAlg::Tensor<T> m_store_Y;
        
        public:
            Parameter<T> weights;

            /// @brief Constructor
            /// @param num_input_nodes Nodes in previous layer
            /// @param nodes Nodes in this layer
            /// @param init The value to initialize all weights to
            Layer(int num_input_nodes, int nodes, T init = 0);

            /// @brief Randomly sets all weights ~U(low, high)
            void uniform(const Rand::Random<T>& random, T low, T high) {
                weights.uniform(random, low, high);
            }

            /// @brief Randomly sets all weights ~N(mean, stddev)
            void normal(const Rand::Random<T>& random, T mean, T stddev) {
                weights.normal(random, mean, stddev);
            }

            /// @brief Performs a forward pass through the layer
            /// @param X Input tensor of shape (1, input nodes)
            /// @return Tensor of shape (1, nodes)
            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            /// @brief Performs backpropagation through the layer
            /// @param dY Backpropagation tensor of shape (1, nodes) from next layer
            /// @return Tensor of shape (1, input size)
            LinAlg::Tensor<T> backward_pass(const LinAlg::Tensor<T>& dY);    
    };

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    Layer<T, F, A>::Layer(int input_nodes, int nodes, T init) 
        : m_nodes {nodes}
        , m_input_nodes {input_nodes}
        , m_store_X {{1, input_nodes}}
        , m_store_Y {{1, nodes}}
        , weights {{nodes, F::num_weights(input_nodes)}, init}
    {}

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    LinAlg::Tensor<T> Layer<T, F, A>::forward_pass(const LinAlg::Tensor<T>& X) {
        LinAlg::Tensor<T> Y {{1, m_nodes}};

        for(int i {}; i < m_nodes; ++i) {
            Y[{0, i}] = F::function(X, weights.value.row(i).unsqueeze());
        }

        m_store_Y = Y;
        m_store_X = X.copy();

        return A::activate(Y);
    }

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    LinAlg::Tensor<T> Layer<T, F, A>::backward_pass(const LinAlg::Tensor<T>& dY) {
        LinAlg::Tensor<T> dZ {LinAlg::pairwise_mult(dY, A::derivate(m_store_Y))};

        for(int i {}; i < m_nodes; ++i) {
            LinAlg::Tensor<T> dW_i {F::weights_grad(m_store_X, weights.value.row(i).unsqueeze())};

            for(int j {}; j < F::num_weights(m_input_nodes); ++j) {
                weights.grad[{i, j}] += dZ[{0, i}] * dW_i[{0, j}];
            }
        }

        LinAlg::Tensor<T> Y {{m_nodes, m_input_nodes}};

        for(int i {}; i < m_nodes; ++i) {
            LinAlg::Tensor<T> dF {F::function_grad(m_store_X, weights.value.row(i).unsqueeze())};

            for(int j {}; j < m_input_nodes; ++j) {
                Y[{i, j}] = dF[{0, j}];
            }
        }

        return dZ * Y;
    }
}  

#endif