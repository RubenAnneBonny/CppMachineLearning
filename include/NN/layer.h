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

            Parameter(const std::vector<int>& shape) 
                : value {shape}
                , grad {shape}
            {}
    };
    
    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    class Layer {
        private:
            int m_num_nodes;
            int m_num_input_nodes;
            int m_num_weights;
            LinAlg::Tensor<T> m_store_input;
            Parameter weights;
        
        public:
            Layer(int num_input_nodes, int nodes);

            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            LinAlg::Tensor<T> backward_pass(const LinAlg::Tensor<T>& X);    
    };

    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    Layer<T, F, A>::Layer(int num_input_nodes, int num_nodes) 
        : m_num_nodes {num_nodes}
        , m_num_input_nodes {num_input_nodes}
        , m_num_weights {num_nodes * F::num_weights(num_input_nodes)}
        , weights {{num_nodes, num_input_nodes}}
    {}
}

#endif