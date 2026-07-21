#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <stdexcept>
#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Func/function.h>
#include <NN/parameter.h>

namespace NN {    
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
            void uniform(Rand::Random<T>& random, T low, T high) {
                weights.uniform(random, low, high);
            }

            /// @brief Randomly sets all weights ~N(mean, stddev)
            void normal(Rand::Random<T>& random, T mean, T stddev) {
                weights.normal(random, mean, stddev);
            }

            int get_nodes() const {
                return m_nodes;
            }

            int get_input_nodes() const {
                return m_input_nodes;
            }

            LinAlg::Tensor<T> get_pre_activation() const {
                return m_store_Y;
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

        int num_weights {F::num_weights(m_input_nodes)};

        for(int i {}; i < m_nodes; ++i) {
            LinAlg::Tensor<T> dW_i {F::weights_grad(m_store_X, weights.value.row(i).unsqueeze())};

            for(int j {}; j < num_weights; ++j) {
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