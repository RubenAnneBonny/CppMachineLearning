#ifndef STD_FUNC_H
#define STD_FUNC_H

#include <Func/function.h>
#include <LinAlg/tensor.h>

namespace Func{
    template <typename T>
    class Linear {
        public:
            static int num_weights(int input_size) {
                return input_size + 1;
            }

            static T function(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                T result {};
                int input_size {X.get_extent(1)};
                
                for(int i {}; i < input_size; ++i) {
                    result += X[{0, i}] * weights[{0, i}];
                }

                return result + weights[{0, input_size}];
            }
    
            static LinAlg::Tensor<T> function_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                int input_size {X.get_extent(1)};
                LinAlg::Tensor<T> grad {{1, input_size}};

                for(int i {}; i < input_size; ++i) {
                    grad[{0, i}] = weights[{0, i}];
                }

                return grad;
            }

            static LinAlg::Tensor<T> weights_grad(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                int input_size {X.get_extent(1)};
                LinAlg::Tensor<T> grad {{1, input_size + 1}};

                for(int i {}; i < input_size; ++i) {
                    grad[{0, i}] = X[{0, i}];
                }

                grad[{0, input_size}] = 1;

                return grad;
            }
    };

    template <typename T>
    class ReLU {
        public:
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                auto relu{
                [](T a)
                {
                    return (a > 0 ? a : 0);
                }
                };

                LinAlg::Tensor<T> activated {X.copy()};
                activated.elementwise(relu);

                return activated;
            }
    
            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                auto relu_grad{
                [](T a)
                {
                    return (a > 0 ? 1 : 0);
                }
                };

                LinAlg::Tensor<T> grad {X.copy()};
                grad.elementwise(relu_grad);

                return grad;
            }
    };

    template <typename T>
    class No_Activation {
        public: 
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                return X.copy();
            }
    
            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                LinAlg::Tensor<T> grad {{1, X.get_extent(1)}, 1};

                return grad;
            }
    };

    template <typename T>
    class MSE {
        public:
            static T loss(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                T loss {};
                int input_size {prediction.get_extent(1)};

                for(int i {}; i < input_size; ++i) {
                    T diff = prediction[{0, i}] - target[{0, i}];
                    loss += diff * diff;
                }    

                loss /= static_cast<T>(input_size);
                
                return loss;
            }
    
            static LinAlg::Tensor<T> gradient(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                int input_size {prediction.get_extent(1)};
                LinAlg::Tensor<T> dL {{1, input_size}};

                for(int i {}; i < input_size; ++i) {
                    dL[{0, i}] = 2 * (prediction[{0, i}] - target[{0, i}]) / static_cast<T>(input_size);
                }

                return dL;
            }
    };
}

#endif