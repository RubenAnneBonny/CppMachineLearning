#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <concepts>
#include <stdexcept>
#include <string>

namespace Rand{
    template<std::floating_point T>
    class Random{
        private:
            std::mt19937 mt;

        public:
            /// @brief Unseeded constructor
            explicit Random()
                : mt {std::random_device{}()}
            {}

            /// @brief Seeded constructor
            /// @param seed The seed
            explicit Random(int seed)
                : mt {static_cast<unsigned int>(seed)}
            {}

            /// @brief Random uniform distrobution
            /// @param low Lower bound of number generation
            /// @param high Upper bound of number generation
            /// @return Random number in range [low, high]
            /// @throws std::invalid_argument if high < low
            T uniform(T low, T high){
                if(high < low) {
                    throw std::invalid_argument(
                        "Cannot produce random number in range [" + 
                        std::to_string(low) + 
                        ", " + 
                        std::to_string(high) + 
                        "] since its an invalid range"
                    );
                }

                return std::uniform_real_distribution<T>{low, high}(mt);
            }

            /// @brief Random normal distrubution
            /// @param mean 
            /// @param stddev Standard deviation
            /// @return Random number ~N(mean, stddev)
            T normal(T mean, T stddev){
                return std::normal_distribution<T>{mean, stddev}(mt);
            }

            /// @brief Random uniform integer distrubution
            /// @param low Lower bound (inclusive)
            /// @param high Upper bound (exclusive)
            /// @return Random integer in range [low, high)
            /// @throws std::invalid_argument if high <= low
            int uniform_int(int low, int high) {
                if(high <= low) {
                    throw std::invalid_argument(
                        "Cannot produce random integer in range [" + 
                        std::to_string(low) + 
                        ", " + 
                        std::to_string(high) + 
                        ") since its an invalid range"
                    );
                }

                return std::uniform_int_distribution<int>{low, high - 1}(mt);
            }
    };
}

#endif