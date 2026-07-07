#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <stdexcept>
#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Func/function.h>

namespace NN {
    template <std::floating_point T,
              Func::Function<T> F,
              Func::Activation_function<T> A>
    class Layer {
        private:
            int m_num_nodes;
            int m_num_input_nodes;
            int m_num_weights;
            std::vector<F> m_nodes;
            A m_activation;
            LinAlg::Tensor<T> m_store_input;
        
        public:
            Layer(int num_input_nodes);

            
    };
}

#endif