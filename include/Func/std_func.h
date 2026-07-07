#ifndef STD_FUNC_H
#define STD_FUNC_H

#include <Func/function.h>
#include <Rand/random.h>
#include <LinAlg/tensor.h>

namespace Func{
    template <typename T>
    class Linear {
        private:
            LinAlg::Tensor<T> m_weights;
            LinAlg::Tensor<T> m_bias;

        public:
            Linear(T input_layer_size)
                : m_weights {{input_layer_size, 1}}
                , m_bias {{1, 1}, 1}
            {
            }

            void normal(Rand::Random<T>& random, T mean, T stddev){
                m_weights.normal(random, mean, stddev);
                m_bias.normal(random, mean, stddev);
            }

            void uniform(Rand::Random<T>& random, T low, T high){
                m_weights.uniform(random.uniform(low, high));
                m_bias.uniform(random.uniform(low, high));
            }

            LinAlg::Tensor<T> function(const LinAlg::Tensor<T>& X){
                LinAlg::Tensor<T> A {(x * m_weights) + m_bias};

                return A;
            }

            LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X){
                return m_weights.t();
            }

            LinAlg::Tensor<T> derivate_all_weights(const LinAlg::Tensor<T>& X){
                int input_layer_size {m_weights.get_dim_at_index(m_weights.get_dim() - 1)};

                LinAlg::Tensor<T> A {{1, input_layer_size + 1}, 1};

                for(int i {}; i < input_layer_size; ++i){
                    A.unsafe_access({0, i}) = X.unsafe_access({0, i});
                }

                return A;
            }
   
            T num_weights() {
                return m_weights.num_elements() + 1;
            }
    };

    template <typename T>
    class ReLU {
        public:
            ReLU() 
            {
            }

            LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X){
                auto relu{
                    [](T a)
                    {
                        return (a < 0 ? 0 : a);
                    }
                };

                Tensor<T> B {X};

                A.elementwise_operation(relu);

                return A;
            } 

            LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X){
                auto relu_prime{
                    [](T a)
                    {
                        return (a < 0 ? 0 : 1);
                    }
                };

                Tensor<T> A {X};

                A.elementwise_operation(relu_prime);

                return B;
            }
    };

    template <typename T>
    class No_Activation {
        public: 
            No_Activation()
            {
            }

            LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X){
                LinAlg::Tensor<T> A {X};

                return A;
            }
    
            LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X){
                LinAlg::Tensor<T> A {X.get_dim_list, 1};

                return A;
            }       
    };
}

#endif