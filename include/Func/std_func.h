#ifndef STD_FUNC_H
#define STD_FUNC_H

#include <Func/function.h>
#include <LinAlg/tensor.h>
#include <cmath>

namespace Func{
    /**
     * @brief Linear node function a * X + b
     */
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
    
            static LinAlg::Tensor<T> input_derivative(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& weights) {
                int input_size {X.get_extent(1)};
                LinAlg::Tensor<T> grad {{1, input_size}};

                for(int i {}; i < input_size; ++i) {
                    grad[{0, i}] = weights[{0, i}];
                }

                return grad;
            }

            static LinAlg::Tensor<T> weight_derivative(const LinAlg::Tensor<T>& X, [[maybe_unused]] const LinAlg::Tensor<T>& weights) {
                int input_size {X.get_extent(1)};
                LinAlg::Tensor<T> grad {{1, input_size + 1}};

                for(int i {}; i < input_size; ++i) {
                    grad[{0, i}] = X[{0, i}];
                }

                grad[{0, input_size}] = 1;

                return grad;
            }
    };

    /**
     * @brief ReLU activation function, performs elementwise max(0, X[element])
     */
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
    
            static LinAlg::Tensor<T> derivative(const LinAlg::Tensor<T>& X) {
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

    /**
     * @brief No_activation activation function that makes it possible to create
     * layers with no activation
     */
    template <typename T>
    class No_activation {
        public: 
            static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
                return X.copy();
            }
    
            static LinAlg::Tensor<T> derivative(const LinAlg::Tensor<T>& X) {
                LinAlg::Tensor<T> grad {{X.get_extent(0), X.get_extent(1)}, 1};

                return grad;
            }
    };

    /**
     * @brief Sigmoid activation function performs elementwise 1 / (1 +
     * e^(-X[element]))
     */
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
    
            static LinAlg::Tensor<T> derivative(const LinAlg::Tensor<T>& X) {
                LinAlg::Tensor<T> activated {activate(X)};

                return activated * (1 - activated);
            }
    };

    /**
     * @brief Mean Squared Error loss function, computes loss as the squared
     * difference between prediction and target
     */
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
                LinAlg::Tensor<T> dL {{1, input_size}};

                for(int i {}; i < input_size; ++i) {
                    dL[{0, i}] = 2 * (prediction[{0, i}] - target[{0, i}]) / static_cast<T>(input_size);
                }

                return dL;
            }
    };

    /**
     * @brief Softmax cross entropy loss function used for classification
     * problems
     * 
     * @note Targets are assumed to be one-hot
     */
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
                int input_size {raw_prediction.get_extent(1)};

                LinAlg::Tensor<T> prediction {{1, input_size}};

                T max_value {raw_prediction.max()};
                T exp_sum {};
                for(int i {}; i < input_size; ++i) {
                    exp_sum += std::exp(raw_prediction[{0, i}] - max_value);
                }

                for(int i {}; i < input_size; ++i) {
                    prediction[{0, i}] = std::exp(raw_prediction[{0, i}] - max_value) / exp_sum;
                }

                return prediction - target;
            }
    };
}

#endif