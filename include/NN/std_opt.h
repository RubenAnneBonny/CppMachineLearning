#ifndef STD_OPT_H
#define STD_OPT_H

#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <LinAlg/tensor.h>
#include <vector>
#include <cmath>

namespace NN {
    template <std::floating_point T>
    class Gradient_descent {
        private:
            T m_lr;
            std::vector<NN::Parameter<T>*> m_parameters;
        
        public:
            explicit Gradient_descent(T lr) 
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

    template <std::floating_point T>
    class Adam {
        private:
            T m_beta_1;
            T m_beta_2;
            T m_alpha;
            T m_eps;
            std::vector<NN::Parameter<T>*> m_parameters;
            std::vector<LinAlg::Tensor<T>> m_first_moment;
            std::vector<LinAlg::Tensor<T>> m_second_moment;
            int m_t;

        public:
            explicit Adam(T beta_1 = 0.9, T beta_2 = 0.999, T alpha = 1e-3, T eps = 1e-8)
                : m_beta_1 {beta_1}
                , m_beta_2 {beta_2}
                , m_alpha {alpha}
                , m_eps {eps}
                , m_first_moment {}
                , m_second_moment {}
                , m_t {}
            {}
            
            void init(std::vector<NN::Parameter<T>*>& parameters) {
                m_parameters = parameters;

                m_first_moment.clear();
                m_second_moment.clear();

                for(int i {}; i < static_cast<int>(m_parameters.size()); ++i) {
                    LinAlg::Tensor<T> X {{m_parameters[i]->value.get_extent(0), m_parameters[i]->value.get_extent(1)}};

                    m_first_moment.push_back(X.copy());
                    m_second_moment.push_back(X.copy());
                }
            }

            void step() {
                ++m_t;

                for(int i {}; i < static_cast<int>(m_parameters.size()); ++i) {
                    auto first{
                        [&](T a, T b)
                        {
                            return m_beta_1 * a + (1 - m_beta_1) * b;
                        }
                    };
                    auto second{
                        [&](T a, T b)
                        {
                            return m_beta_2 * a + (1 - m_beta_2) * b * b;
                        }
                    };

                    m_first_moment[i] = LinAlg::pairwise<T>(m_first_moment[i], m_parameters[i]->grad, first);
                    m_second_moment[i] = LinAlg::pairwise<T>(m_second_moment[i], m_parameters[i]->grad, second);

                    LinAlg::Tensor<T> first_hat = m_first_moment[i] * (1 / (1 - std::pow(m_beta_1, m_t)));
                    LinAlg::Tensor<T> second_hat = m_second_moment[i] * (1 / (1 - std::pow(m_beta_2, m_t)));

                    auto step_size {
                        [&](T a, T b)
                        {
                            return m_alpha * a / (std::sqrt(b) + m_eps);
                        }
                    };

                    m_parameters[i]->value -= LinAlg::pairwise<T>(first_hat, second_hat, step_size);
                }
            }
    };
}

#endif