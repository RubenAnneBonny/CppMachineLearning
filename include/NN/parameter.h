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

            Parameter(const Parameter& other) 
                : value {other.value.copy()}
                , grad {other.grad.copy()}
            {}

            void uniform(Rand::Random<T>& random, T low, T high) {
                value.uniform(random, low, high);
            }

            void normal(Rand::Random<T>& random, T mean, T stddev) {
                value.normal(random, mean, stddev);
            }

            void zero_grad() {
                grad.set_all_elements(0);
            }

            Parameter& operator=(const Parameter& other) {
                value = other.value.copy();
                grad = other.grad.copy();
                return *this;
            }

            Parameter(Parameter&&) = default;
            Parameter& operator=(Parameter&&) = default;
    };
}

#endif