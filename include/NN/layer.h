#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <stdexcept>
#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Func/function.h>
#include <NN/parameter.h>

namespace NN {    
    /**
     * @brief Defines forward and backward passes to use when propagating
     * through the neural network
     *
     * @details Contains the nodes for a layer, saving states for input tensor
     * and output tensor in forward_pass to use in backward_pass
     * 
     * @code
     * NN::Layer<double, Func::Linear<double>, Func::ReLU<double>> layer {2, 3};
     * LinAlg::Tensor<double> X {{1, 2}};
     * LinAlg::Tensor<double> output {layer.forward_pass(X)};
     * @endcode
     */
    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    class Layer {
        private:
            int m_nodes;
            int m_input_nodes;
            LinAlg::Tensor<T> m_store_X;
            LinAlg::Tensor<T> m_store_Y;

            LinAlg::Tensor<T> forward(const LinAlg::Tensor<T>& X) const {
                int batches {X.get_extent(0)};
                LinAlg::Tensor<T> Y {{batches, m_nodes}};

                for(int b {}; b < batches; ++b) {
                    LinAlg::Tensor<T> X_b {X.row(b).unsqueeze()};

                    for(int i {}; i < m_nodes; ++i) {
                        Y[{b, i}] = F::function(X_b, weights.value.row(i).unsqueeze());
                    }
                }

                return Y;
            }
        
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
                return m_store_Y.copy();
            }

            /// @brief Performs a forward pass through the layer
            /// @param X Input tensor of shape (Batch, input nodes)
            /// @return Tensor of shape (Batch, nodes)
            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            /// @brief Performs a forward pass through the layer, does not save internal states
            /// @param X Input tensor of shape (Batch, input nodes)
            /// @return Tensor of shape (Batch, nodes)
            LinAlg::Tensor<T> forward_pass_stateless(const LinAlg::Tensor<T>& X) const;

            /// @brief Performs backpropagation through the layer
            /// @param dY Backpropagation tensor of shape (Batch, nodes) from next layer
            /// @return Tensor of shape (Batch, input size)
            /// @throws std::invalid_argument if batch dimension of dY and m_store_X don't match
            /// @throws std::invalid_argument if extent of second axis of dY don't match m_nodes
            LinAlg::Tensor<T> backward_pass(const LinAlg::Tensor<T>& dY);    
    };

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    Layer<T, F, A>::Layer(int input_nodes, int nodes, T init) 
        : m_nodes {nodes}
        , m_input_nodes {input_nodes}
        , m_store_X {{1, input_nodes}}
        , m_store_Y {{1}}
        , weights {{nodes, F::num_weights(input_nodes)}, init}
    {}

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    LinAlg::Tensor<T> Layer<T, F, A>::forward_pass(const LinAlg::Tensor<T>& X) {
        LinAlg::Tensor<T> Y {forward(X)};

        m_store_Y = Y;
        m_store_X = X.copy();

        return A::activate(Y);
    }

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    LinAlg::Tensor<T> Layer<T, F, A>::forward_pass_stateless(const LinAlg::Tensor<T>& X) const {
        LinAlg::Tensor<T> Y {forward(X)};

        return A::activate(Y);
    }    

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    LinAlg::Tensor<T> Layer<T, F, A>::backward_pass(const LinAlg::Tensor<T>& dY) {
        if(dY.get_extent(1) != m_nodes) {
            throw std::invalid_argument(
                "Cannot perform backward pass with tensor dY of shape " + 
                static_cast<std::string>(dY) + 
                " since last axis must match number of nodes in layer which is " + 
                std::to_string(m_nodes)
            );
        }

        if(dY.get_extent(0) != m_store_X.get_extent(0)) {
            throw std::invalid_argument(
                "Cannot perform backward pass with tensor dY of shape " + 
                static_cast<std::string>(dY) + 
                " since its batch dimension dont match the stored x tensor from forward pass of shape " + 
                static_cast<std::string>(m_store_X)
            );
        }
        
        int batches {dY.get_extent(0)};
        LinAlg::Tensor<T> dZ {dY * A::derivative(m_store_Y)};

        int num_weights {F::num_weights(m_input_nodes)};

        LinAlg::Tensor<T> dX {{batches, m_input_nodes}};

        for(int b {}; b < batches; ++b) {
            LinAlg::Tensor<T> X_b {m_store_X.row(b).unsqueeze()};

            for(int i {}; i < m_nodes; ++i) {
                LinAlg::Tensor<T> weight_i {weights.value.row(i).unsqueeze()};

                LinAlg::Tensor<T> weight_derivative {F::weight_derivative(X_b, weight_i)};
                for(int j {}; j < num_weights; ++j) {
                    weights.grad[{i, j}] += dZ[{b, i}] * weight_derivative[{0, j}];
                }

                LinAlg::Tensor<T> input_derivative {F::input_derivative(X_b, weight_i)};
                for(int j {}; j < m_input_nodes; ++j) {
                    dX[{b, j}] += dZ[{b, i}] * input_derivative[{0, j}];
                }
            }
        }

        return dX;
    }
}  

#endif