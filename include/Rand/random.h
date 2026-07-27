#ifndef CML_RANDOM_H
#define CML_RANDOM_H

#include <random>
#include <concepts>
#include <stdexcept>
#include <string>
#include <vector>

namespace Rand{
    /**
     * @brief A random class used for creating random numbers
     */
    template<std::floating_point T>
    class Random{
        private:
            std::mt19937 m_mt;

        public:
            /// @brief Unseeded constructor
            explicit Random()
                : m_mt {std::random_device{}()}
            {}

            /// @brief Seeded constructor
            /// @param seed The seed
            explicit Random(int seed)
                : m_mt {static_cast<unsigned int>(seed)}
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

                return std::uniform_real_distribution<T>{low, high}(m_mt);
            }

            /// @brief Random normal distrubution
            /// @param mean 
            /// @param stddev Standard deviation
            /// @return Random number ~N(mean, stddev)
            T normal(T mean, T stddev){
                return std::normal_distribution<T>{mean, stddev}(m_mt);
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

                return std::uniform_int_distribution<int>{low, high - 1}(m_mt);
            }

            /// @brief A Fisher-Yates shuffled permutation of [0, length)
            /// @param length The length of the permutation
            /// @return The permutation
            /// @throws std::invalid_argument if lenght is less than 0
            std::vector<int> permutation(int length) {
                if(length < 0) {
                    throw std::invalid_argument(
                        "Cannot create permutation of length " + 
                        std::to_string(length) + 
                        " since its a negative number"
                    );
                }

                std::vector<int> p(static_cast<std::size_t>(length));
                for(int i {}; i < length; ++i) {
                    p[static_cast<std::size_t>(i)] = i;
                }

                for(int i {length - 1}; i > 0; --i) {
                    int j {uniform_int(0, i + 1)};
                    std::swap(p[static_cast<std::size_t>(i)], p[static_cast<std::size_t>(j)]);
                }

                return p;
            }
    };
}

#endif