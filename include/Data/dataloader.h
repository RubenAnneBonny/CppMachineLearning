#ifndef DATALOADER_H
#define DATALOADER_H

#include <LinAlg/tensor.h>
#include <Rand/random.h>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace Data {
    template <std::floating_point T>
    class Dataloader {
        private:
            const LinAlg::Tensor<T> m_inputs;
            const LinAlg::Tensor<T> m_targets;
            std::vector<int> m_permutation;
            int m_batch;
            int m_batch_size;

            void shuffle(Rand::Random<T>& random) {
                for(int i {static_cast<int>(m_permutation.size()) - 1}; i > 0; --i) {
                    int j {random.uniform_int(0, i + 1)};
                    std::swap(m_permutation[i], m_permutation[j]);
                }
            }

        public:
            /// @brief A constructor for the dataloader
            /// @param random A random instance to shuffle the permutation
            /// @param inputs The tensor to batch as input
            /// @param targets The tensor to batch as target
            /// @param batch_size The batch size to use when loading
            /// @throws std::invalid_argument if batch_size is less than 1
            /// @throws std::invalid_argument if the extents of the first axises of targets and inputs don't match
            Dataloader(Rand::Random<T>& random, const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int batch_size);
       
            int get_num_batches(bool drop_last = false) {
                if(drop_last) {
                    return (static_cast<int>(m_permutation.size()) / m_batch_size);
                }
                return ((static_cast<int>(m_permutation.size()) + m_batch_size - 1) / m_batch_size);
            }

            /// @brief Gets the next batched inputs and targets
            /// @param random A random instance to shuffle the permutation
            /// @param inputs The tensor to put the batched inputs in
            /// @param targets The tensor to put the batched targets in
            /// @param drop_last If last batch will be incomplete, drop it or not
            /// @return True if it hasn't iterated through the all smamples, false if it has
            bool next_batch(Rand::Random<T>& random, LinAlg::Tensor<T>& inputs, LinAlg::Tensor<T>& targets, bool drop_last = false);
    };

    template <std::floating_point T>
    Dataloader<T>::Dataloader(Rand::Random<T>& random, const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int batch_size) 
        : m_inputs {inputs}
        , m_targets {targets}
        , m_permutation(inputs.get_extent(0))
        , m_batch {}
        , m_batch_size {batch_size}
    {        
        if(batch_size < 1) {
            throw std::invalid_argument(
                "Cannot get next batch, if batch size is 0"
            );
        }

        if(inputs.get_extent(0) != targets.get_extent(0)) {
            throw std::invalid_argument(
                "The extents of the first axises must match on input and target tensors, currently inputs has shape " + 
                static_cast<std::string>(inputs) + 
                " and targets have shape " + 
                static_cast<std::string>(targets)
            );
        }

        for(int i {}; i < static_cast<int>(m_permutation.size()); ++i) {
            m_permutation[i] = i;
        }
        shuffle(random);
    }

    template <std::floating_point T>
    bool Dataloader<T>::next_batch(Rand::Random<T>& random, LinAlg::Tensor<T>& X, LinAlg::Tensor<T>& Y, bool drop_last) {        
        if(m_batch == get_num_batches(drop_last)){
            shuffle(random);
            m_batch = 0;

            return false;
        }

        int lower_index {m_batch * m_batch_size};
        int upper_index {std::min((m_batch + 1) * m_batch_size, static_cast<int>(m_permutation.size()))};

        std::vector<int> indecies(m_permutation.begin() + lower_index, m_permutation.begin() + upper_index);
        X = m_inputs.gather(indecies);
        Y = m_targets.gather(indecies);
        
        ++m_batch;
        return true;
    }
}   

#endif