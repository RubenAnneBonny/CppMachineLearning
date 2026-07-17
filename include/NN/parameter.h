#ifndef PARAMETER_H
#define PARAMETER_H

#include <LinAlg/tensor.h>

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
}

#endif