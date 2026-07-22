#ifndef DATALOADER_H
#define DATALOADER_H

#include <LinAlg/tensor.h>
#include <Rand/random.h>

namespace Data {
    template <std::floating_point T>
    class Dataloader {
        private:
            Rand::Random<T>& m_random;
            const LinAlg::Tensor<T>& m_input;
            const LinAlg::Tensor<T>& m_target;
            std::vector<int> m_permutation;
            int m_batch;

            void shuffle() {
                for(int i {static_cast<int>(m_permutation.size()) - 1}; i > 0; --i) {
                    int j {static_cast<int>(m_random.uniform(0, i + 1))};
                    std::swap(m_permutation[i], m_permutation[j]);
                }
            }

        public:
            Dataloader(Rand::Random<T>& random, const LinAlg::Tensor<T>& input, const LinAlg::Tensor<T>& target);
       
            int get_num_batches(int batch_size) {
                return ((static_cast<int>(m_permutation.size()) + batch_size - 1) / batch_size);
            }

            bool next_batch(LinAlg::Tensor<T>& input, LinAlg::Tensor<T>& target, int batch_size);
    };

    template <std::floating_point T>
    Dataloader<T>::Dataloader(Rand::Random<T>& random, const LinAlg::Tensor<T>& input, const LinAlg::Tensor<T>& target) 
        : m_random {random}
        , m_input {input}
        , m_target {target}
        , m_permutation(input.get_extent(0))
        , m_batch {}
    {
        for(int i {}; i < static_cast<int>(m_permutation.size()); ++i) {
            m_permutation[i] = i;
        }
        shuffle();
    }

    template <std::floating_point T>
    bool Dataloader<T>::next_batch(LinAlg::Tensor<T>& input, LinAlg::Tensor<T>& target, int batch_size) {
        if(m_batch == get_num_batches(batch_size)){
            shuffle();
            m_batch = 0;

            return false;
        }

        int lower_index {m_batch * batch_size};
        int upper_index {std::min((m_batch + 1) * batch_size, static_cast<int>(m_permutation.size()))};

        std::vector<int> indecies(m_permutation.begin() + lower_index, m_permutation.begin() + upper_index);
        input = m_input.gather(indecies);
        target = m_target.gather(indecies);
        
        ++m_batch;
        return true;
    }
}   

#endif