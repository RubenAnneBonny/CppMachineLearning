#ifndef CML_DATA_GENERATION_H
#define CML_DATA_GENERATION_H

#include <LinAlg/tensor.h>
#include <Rand/random.h>
#include <numbers>
#include <cmath>

namespace Data {
    /// @brief Pairs an input tensor to a target tensor
    /// @tparam A floating point type
    template <std::floating_point T>
    class Data_set {
        public:
            LinAlg::Tensor<T> inputs;
            LinAlg::Tensor<T> targets;

            /// @brief Constructor for the data set
            /// @param X The input tensor
            /// @param Y The target tensor
            /// @throws std::invalid_argument if X and Y are of different rank
            /// @throws std::invalid_argument if the batch dimension of X and Y are different
            explicit Data_set(const LinAlg::Tensor<T>& X, const LinAlg::Tensor<T>& Y) 
                : inputs {X}
                , targets {Y}
            {
                if(X.get_rank() != Y.get_rank()) {
                    throw std::invalid_argument(
                        "Cannot create a data set of tensors of shape " + 
                        static_cast<std::string>(X) + 
                        " and " + 
                        static_cast<std::string>(Y) + 
                        " since they are of different ranks"
                    );
                }

                if(X.get_extent(0) != Y.get_extent(0)) {
                    throw std::invalid_argument(
                        "Cannot create a data set of tensors of shape " + 
                        static_cast<std::string>(X) + 
                        " and " + 
                        static_cast<std::string>(Y) +
                        " since their batch dimensions do not match"
                    );
                }
            }

            /// @brief Shuffles the batch dimension of the data set
            /// @param random A random instance
            void shuffle(Rand::Random<T>& random) {
                std::vector<int> permutation {random.permutation(inputs.get_extent(0))};
                inputs = inputs.gather(permutation);
                targets = targets.gather(permutation);
            }
    };

    /// @brief Classification, makes dots on two circles with different radius + noise
    /// @tparam T A floating point type
    /// @param random A random instance
    /// @param num_samples The number of samples to create
    /// @param noise The amount of noise in the data
    /// @param radius_factor How much larger the second circle is than the first
    /// @return A data_set with a input tensor of 2D-coordinates, shape
    /// (num_samples, 2), and one-hot targets, shape (num_samples, 2)
    template <std::floating_point T>
    Data_set<T> make_circles(Rand::Random<T>& random, int num_samples, T noise = T{0.05}, T radius_factor = T{0.5}) {
        LinAlg::Tensor<T> X {{num_samples, 2}};
        LinAlg::Tensor<T> Y {{num_samples, 2}};

        for(int i {}; i < num_samples; ++i) {
            int cls {random.uniform_int(0, 2)};
            T radius {cls ? radius_factor : T{1}};
            T theta {random.uniform(T{0}, T{2} * std::numbers::pi_v<T>)};
            
            X[{i, 0}] = radius * std::cos(theta) + random.normal(T{0}, noise);
            X[{i, 1}] = radius * std::sin(theta) + random.normal(T{0}, noise);
            
            Y[{i, cls}] = T{1};
        }

        Data_set<T> data_set {X, Y};
        data_set.shuffle(random);

        return data_set;
    }

    /// @brief Classification, makes dots on two inteviened half-moons
    /// @tparam T A floating point type
    /// @param random A random instance
    /// @param num_samples The amount of samples to create
    /// @param noise The amound of noise in the data
    /// @return A data_set with input tensor 2D-coordinates, shape (num_samples, 2),
    /// and one-hot targets, shape (num_samples, 2)
    template <std::floating_point T>
    Data_set<T> make_moons(Rand::Random<T>& random, int num_samples, T noise = T{0.1}) {
        LinAlg::Tensor<T> X {{num_samples, 2}};
        LinAlg::Tensor<T> Y {{num_samples, 2}};

        for(int i {}; i < num_samples; ++i) {
            int cls = random.uniform_int(0, 2);
            T theta = random.uniform(T{0}, std::numbers::pi_v<T>);
            
            if(cls == 0) {
                X[{i, 0}] = std::cos(theta) + random.normal(T{0}, noise);
                X[{i, 1}] = std::sin(theta) + random.normal(T{0}, noise);
            }
            else{
                X[{i, 0}] = T{1} - std::cos(theta) + random.normal(T{0}, noise);
                X[{i, 1}] = T{0.5} - std::sin(theta) + random.normal(T{0}, noise);
            }

            Y[{i, cls}] = T{1};
        }

        Data_set<T> data_set {X, Y};
        data_set.shuffle(random);

        return data_set;
    }
    
    /// @brief Classification, makes dots on several spirals
    /// @tparam T A floating point type
    /// @param random A random instance
    /// @param num_samples_per_class The amount of samples to create per class
    /// @param classes The amount of classes/spirals to create
    /// @param noise The amount of noise in the data
    /// @return A data_set with input tensor of 2D-coordinates, shape
    /// (num_samples_per_class * classes, classes), and one-hot targets, shape
    /// (num_samples_per_target * classes, classes)
    template <std::floating_point T>
    Data_set<T> make_spirals(Rand::Random<T>& random, int num_samples_per_class, int classes = 3, T noise = T{0.2}) {
        int num_samples {num_samples_per_class * classes};
        LinAlg::Tensor<T> X {{num_samples, 2}};
        LinAlg::Tensor<T> Y {{num_samples, classes}};

        int row {};
        for(int cls {}; cls < classes; ++cls) {
            for(int i {}; i < num_samples_per_class; ++i) {
                T fraction {static_cast<T>(i) / static_cast<T>(num_samples_per_class)};
                T radius = fraction;
                T theta = fraction * T{3} + (T{2} * std::numbers::pi_v<T> * static_cast<T>(cls)) / static_cast<T>(classes) + random.normal(T{0}, noise);

                X[{row, 0}] = radius * std::sin(theta);
                X[{row, 1}] = radius * std::cos(theta);

                Y[{row, cls}] = 1;
                ++row;
            }
        }

        Data_set<T> data_set {X, Y};
        data_set.shuffle(random);

        return data_set;
    }

    /// @brief Calculates the call price using Black-Scholes
    /// @tparam T A floating point type
    /// @param S The spot, current asset price
    /// @param K The strike
    /// @param time_to_expiry Time until the options expires 
    /// @param sigma The volatility of the asset
    /// @param r The risk free rent annualized
    /// @return The call price
    template <std::floating_point T>
    T black_scholes_call(T S, T K, T time_to_expiry, T sigma, T r) {
        T sqrt_t {std::sqrt(time_to_expiry)};
        T d1 {(std::log(S / K) + (r + T{0.5} * sigma * sigma) * time_to_expiry) / (sigma * sqrt_t)};
        T d2 {d1 - sigma * sqrt_t};

        T inv_sqrt_2 {static_cast<T>(0.70710678118654752440)};
        auto norm_cdf{
            [inv_sqrt_2](T x)
            {
                return T{0.5} * (T{1} + std::erf(x * inv_sqrt_2));
            }
        };

        return S * norm_cdf(d1) - K * std::exp(-r * time_to_expiry) * norm_cdf(d2);
    } 

    /// @brief Creates a data set for estimation of black-scholes pricing
    /// @tparam T A floating point type
    /// @param random A random instance
    /// @param num_samples The amount of samples to create
    /// @return A data set with inputs [S / K, time_to_expiry, sigma, r], shape
    /// (num_samples, 4), and targets the call price, shape (num_samples, 1)
    template <std::floating_point T>
    Data_set<T> make_black_scholes(Rand::Random<T>& random, int num_samples) {
        LinAlg::Tensor<T> X {{num_samples, 4}};
        LinAlg::Tensor<T> Y {{num_samples, 1}};

        T S {T{1}};

        for(int i {}; i < num_samples; ++i) {
            T K {random.uniform(T{0.5}, T{1.5})};
            T time_to_expiry {random.uniform(T{0.05}, T{2})};
            T sigma {random.uniform(T{0.1}, T{0.6})};
            T r {random.uniform(T{0}, T{0.08})};

            T price {black_scholes_call<T>(S, K, time_to_expiry, sigma, r)};

            X[{i, 0}] = S / K;
            X[{i, 1}] = time_to_expiry;
            X[{i, 2}] = sigma;
            X[{i, 3}] = r;

            Y[{i, 0}] = price;
        }

        Data_set<T> data_set {X, Y};
        data_set.shuffle(random);

        return data_set;
    }
}

#endif