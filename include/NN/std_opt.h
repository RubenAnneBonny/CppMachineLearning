#ifndef STD_OPT_H
#define STD_OPT_H

#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <LinAlg/tensor.h>
#include <vector>

namespace NN {
    template <typename T>
    class Gradient_descent {
        private:
            T m_lr;
            std::vector<NN::Parameter<T>*> m_parameters;
        
        public:
            Gradient_descent(T lr) 
                : m_lr {lr}
            {}

            void init(std::vector<NN::Parameter<T>*>& parameters) {
                m_parameters = parameters;
            }

            void step() {
                for(auto* param : m_parameters) {
                    param->value -= param->grad * m_lr;
                }
            }
    };
}

#endif