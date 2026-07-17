#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <stdexcept>
#include <concepts>
#include <LinAlg/tensor.h>
#include <NN/parameter.h>

namespace NN {
    template <typename O, typename T>
    concept Optimizer = 
        requires(O o, std::vector<NN::Parameter<T>*>& parameters) {
            /// @brief Saves parameters internally and does neccessary initialization
            {o.init(parameters)};
            /// @brief Updates the value of all parameters
            {o.step()};
        };
}

#endif