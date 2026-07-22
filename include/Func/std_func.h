#ifndef STD_FUNC_H
#define STD_FUNC_H

#include <Func/function.h>
#include <LinAlg/tensor.h>
#include <cmath>

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
                LinAlg::Tensor<T> grad {{X.get_extent(0), X.get_extent(1)}, 1};

                return grad;
            }
    };

    template <typename T>
    class Sigmoid {
        public: 
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                auto sigmoid{
                    [](T a)
                    {
                        return T{1.0} / (T{1.0} + std::exp(-a));
                    }
                };

                LinAlg::Tensor<T> activated {X.copy()};
                activated.elementwise(sigmoid);

                return activated;
            }
    
            static LinAlg::Tensor<T> derivate(const LinAlg::Tensor<T>& X) {
                LinAlg::Tensor<T> activated {activate(X)};

                return LinAlg::pairwise_mult<T>(activated, (1 - activated));
            }
    };

    template <typename T>
    class MSE {
        public:
            static T loss(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                T loss {};
                int input_size {prediction.get_extent(1)};
                int batches {prediction.get_extent(0)};

                for(int b {}; b < batches; ++b) {
                    T batch_loss {};

                    for(int i {}; i < input_size; ++i) {
                        T diff = prediction[{b, i}] - target[{b, i}];
                        batch_loss += diff * diff;
                    }  

                    batch_loss /= static_cast<T>(input_size);
                    loss += batch_loss;
                }  

                loss /= static_cast<T>(batches);
                
                return loss;
            }
    
            static LinAlg::Tensor<T> gradient(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                int input_size {prediction.get_extent(1)};
                int batches {prediction.get_extent(0)};
                LinAlg::Tensor<T> dL {{batches, input_size}};

                for(int b {}; b < batches; ++b) {
                    for(int i {}; i < input_size; ++i) {
                        dL[{b, i}] = 2 * (prediction[{b, i}] - target[{b, i}]) / static_cast<T>(input_size * batches);
                    }
                }

                return dL;
            }
    };

    template <typename T>
    class Softmax_cross_entropy {
        public:
            static T loss(const LinAlg::Tensor<T>& prediction, const LinAlg::Tensor<T>& target) {
                int input_size {prediction.get_extent(1)};
                int batches {prediction.get_extent(0)};

                T loss {};

                for(int b {}; b < batches; ++b) {
                    T max_value {prediction.row(b).max()};
                    T exp_sum {};
                    for(int i {}; i < input_size; ++i) {
                        exp_sum += std::exp(prediction[{b, i}] - max_value);
                    }

                    std::vector<int> class_indecies {target.row(b).unsqueeze().argmax()};
                    class_indecies[0] = b;

                    loss += -(prediction[class_indecies] - max_value) + std::log(exp_sum);
                }

                loss /= static_cast<T>(batches);

                return loss;
            }
            
            static LinAlg::Tensor<T> gradient(const LinAlg::Tensor<T>& raw_prediction, const LinAlg::Tensor<T>& target) {
                int batches {raw_prediction.get_extent(0)};
                int input_size {raw_prediction.get_extent(1)};

                LinAlg::Tensor<T> prediction {{batches, input_size}};

                for(int b {}; b < batches; ++b) {
                    T max_value {raw_prediction.row(b).max()};
                    T exp_sum {};
                    for(int i {}; i < input_size; ++i) {
                        exp_sum += std::exp(raw_prediction[{b, i}] - max_value);
                    }

                    for(int i {}; i < input_size; ++i) {
                        prediction[{b, i}] = std::exp(raw_prediction[{b, i}] - max_value) / exp_sum;
                    }
                }

                return (T{1} / static_cast<T>(batches)) * (prediction - target);
            }
    };
}

#endif