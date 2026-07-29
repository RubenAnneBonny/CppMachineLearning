#ifndef CML_OPTIMIZER_H
#define CML_OPTIMIZER_H

#include <vector>
#include <stdexcept>
#include <concepts>
#include <LinAlg/tensor.h>
#include <NN/parameter.h>

namespace NN {
    /**
     * @brief Defines the concept for an optimizer
     *
     * @details Init should save parameters internally and do necessary
     * initialization. Step should update the values of all parameters
     *
     * @note init should reset all member variables, do not assume init is only
     * run once
     *
     * @code
     * template <std::floating_point T> class My_optimizer { private:
     * std::vector<NN::Parameter<T>*> m_params;
     *
     *      public:
     *          My_optimizer()  
     *              : m_params {}
     *          {}
     *
     *          void init(std::vector<NN::Parameter<T>*>& parms) {
     *              m_params = params;
     *          }
     *
     *          void step() {
     *              for(auto* param : m_params) {
     *                  param->value -= param->grad;
     *              }
     *          }
     * };
     * @endcode
     */
    template <typename O, typename T>
    concept Optimizer = 
        requires(O o, std::vector<NN::Parameter<T>*>& parameters) {
            {o.init(parameters)};
            {o.step()};
        };
}

#endif