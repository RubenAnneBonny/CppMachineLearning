#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <concepts>

namespace Rand{
    template<std::floating_point T>
    class Random{
        private:
            std::mt19937 mt;

        public:
            /// @brief Unseeded constructor
            Random()
                : mt {std::random_device{}()}
            {
            }

            /// @brief Seeded constructor
            /// @param seed The seed
            Random(int seed)
                : mt {static_cast<unsigned int>(seed)}
            {
            }

            /// @brief Random uniform distrobution
            /// @param low Lower bound of number generation
            /// @param high Upper bound of number generation
            /// @return Random number in range [low, high]
            T uniform(T low, T high){
                return std::uniform_real_distribution<T>{low, high}(mt);
            }

            /// @brief Random normal distrubution
            /// @param mean 
            /// @param stddev Standard deviation
            /// @return Random number ~N(mean, stddev)
            T normal(T mean, T stddev){
                return std::normal_distribution<T>{mean, stddev}(mt);
            }
    };
}

#endif