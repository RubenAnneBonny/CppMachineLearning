#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <Rand/random.h>

namespace LinAlg {
    template <std::floating_point T>
    class Tensor;

    template <std::floating_point T, typename Fn>
    Tensor<T> pairwise(const Tensor<T>& A, const Tensor<T>& B, Fn fn);

    template <std::floating_point T>
    Tensor<T> pairwise_mult(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    Tensor<T> pairwise_add(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    class Tensor {
        private:
            std::vector<int> m_shape;
            std::shared_ptr<std::vector<T>> m_storage;
            int m_offset;
            std::vector<int> m_strides;

            int get_rank() const {
                return static_cast<int>(shape.size());
            }

            int num_elements() const {
                int elements {1};

                for(int i {}; i < get_dim(); ++i) {
                    elements *= m_shape[i];
                }

                return elements;
            }

            void calculate_strides() {
                m_strides.clear();
                m_strides.assign(num_elements(), 1);

                for(int j {get_dim() - 2}; j >= 0; ++j) {
                    m_strides[j] = m_strides[j + 1] * m_shape[j + 1];
                }
            }

        public:
            Tensor(const std::vector<int>& shape, T init);

            void normal(Rand::Random<T>& random, T mean, T stddev);

            void uniform(Rand::Random<T>& random, T low, T high);
            
            Tensor copy() const;

            Tensor row(int i) const;

            void unsqueeze(int axis = 0);

            void squeeze(int axis = 0);

            Tensor t() const;

            template <typename Fn>
            void elementwise(Fn fn);

            template <typename Fn>
            friend Tensor<t> pairwise<T, Fn>(const Tensor<T>& A, const Tensor<T>& B, Fn fn);

            operator std::string() const;

            const T& operator[](const std::vector<int>& indecies) const;
            T& operator[](const std::vector<int>& indecies);

            friend Tensor operator*(const Tensor& A, const Tensor& B);
            Tensor& operator*=(const Tensor& A);

            friend bool operator==(const Tensor& A, const Tensor& B);
            friend bool operator!=(const Tensor& A, const Tensor& B);

            friend Tensor operator+(const Tensor& A, const Tensor& B);
            friend Tensor<T> pairwise_add<T>(const Tensor<T>& A, const Tensor<T>& B);

            friend Tensor operator+(const Tensor& A, T b);
            friend Tensor operator+(T b, const Tensor& A);
            Tensor& operator+=(T a);

            friend Tensor<T> pairwise_mult<T>(const Tensor<T>& A, const Tensor<T>& B);

            friend Tensor operator*(const Tensor& A, T b);
            friend Tensor operator*(T b, const Tensor& A);
            Tensor& operator*=(T b);
    };

    template <std::floating_point T>
    Tensor<T>::Tensor(const std::vector<int>& shape, T init)
        : m_shape {shape}
        , m_storage {std::make_shared<std::vector<T>>(num_elements(), init)}
        , m_offset {}
        , m_strides {calculate_strides()}
    {}

    template <std::floating_point T>
    void Tensor<T>::normal(Rand::Random<T>& random, T mean, T stddev) {
        for(int i {m_offset}; i < num_elements(); ++i) {
            m_storage[i] = random.normal(mean, stddev);
        }
    }

    template <std::floating_point T>
    void Tensor<T>::uniform(Rand::Random<T>& random, T low, T high) {
        for(int i {m_offset}; i < num_elements(); ++i) {
            m_storage[i] = random.uniform(low, high);
        }
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::copy() const {
        Tensor<T> A {m_shape};
        
        for(int i {m_offset}; i < num_elements(); ++i){
            A.m_storage[i - m_offset] = m_storage[i];
        }

        return A;
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::row(int i) const {
        if(get_rank() == 1){
            throw std::invalid_argument(
                "Cannot use .row on Tensor with only 1 dimension"
            );
        }

        Tensor A {*this};
        A.m_offset += i * m_strides[0];
        A.m_shape.erase(A.m_shape.begin());
        A.m_strides.erase(A.m_strides.begin());

        return A;
    }

    template <std::floating_point T>
    void Tensor<T>::unsqueeze(int axis) {
        int strides {get_rank()};

        if(axis > get_rank() || axis < 0){
            "Cannot unsqueeze Tensor of shape " +
            static_cast<std::string>(*this) + 
            " in axis " + 
            std::to_string(axis) + 
            " since its outside the tensors rank"
        }

        if(axis > 0) {
            strides = m_strides[axis - 1]; 
        }

        m_shape.insert(m_shape.begin() + axis);
        m_strides.insert(m_strides.begin() + axis);
    }

    template <std::floating_point T>
    void Tensor<T>::squeeze(int axis) {
        if(axis >= get_rank() || axis < 0){
            throw std::invalid_argument(
                "Cannot squeeze Tensor of shape " +
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since its outisde its rank"
            );
        }

        if(m_shape[axis] != 1){
            throw std::invalid_argument(
                "Cannot squeeze Tensor of shape " + 
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since the extent of that axis is not 1"
            );
        }

        m_shape.erase(m_shape.begin() + axis);
        m_strides.erase(m_strides.begin() + axis);
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::t() const {
        int rank {get_rank()};

        if(rank < 2){
            throw std::invalid_argument(
                "Cannot transpose Tensor of shape " + 
                static_cast<std::string>(*this) + 
                " since its rank is less than 2"
            );
        }

        Tensor A {*this};

        int temp {A.m_shape[rank - 1]};
        A.m_shape[rank - 1] = A.m_shape[rank - 2];
        A.m_shape[rank - 2] = temp;
    }

    template <std::floating_point T>
    template <typename Fn>
    void Tensor<T>::elementwise(Fn fn) {
        for(int i {m_offset}; i < num_elements(); ++i) {
            m_storage[i] = fn(m_storage);
        }
    }

    template <std::floating_point T, typename Fn>
    Tensor<T> pairwise(const Tensor<T>& A, const Tensor<T>& B, Fn fn) {
        Tensor<T> A_view {A};
        Tensor<T> B_view {B};

        int A_rank {A.get_rank()};
        int B_rank {B.get_rank()};

        int rank {A_rank > B_rank ? A_rank : B_rank};

        Tensor<T> C;

        {
            Tensor<T>& max_ref {A_rank == rank ? A_view : B_view};
            Tensor<T>& min_ref {A_rank == rank ? B_view : A_view};

            int min_rank {min_ref.get_rank()};

            for(int i {min_rank - 1}, j {rank - 1}; i >= 0; --i, --j) {
                if(max_ref.m_shape[j] != min_ref.m_shape[i]){
                    throw std::invalid_argument(
                        "Cannot perform a pairwise operation on two tensors of shape" + 
                        static_cast<std::string>(A) + 
                        " and " + 
                        static_cast<std::string>(B) + 
                        " the extents of their existing dimensions must match"
                    );
                }
            }

            std::vector<int> strides(rank - min_rank, 0);
            for(int i {}; i < min_rank; ++i){
                strides.push_back(min_ref.m_strides[i]);
            }

            min_ref.m_strides = strides;
            min_ref.m_shape = max_ref.m_shape;

            C = Tensor<T>{max_ref.m_shape};
        }

        std::vector<int> indecies(rank, 0);

        for(int i {}; i < C.num_elements(); ++i) {
            C[indecies] = fn(A_view[indecies], B_view[indecies]);

            int index {rank - 1};

            while(++indecies[index] == A_view.m_shape[index]){
                indecies[index] = 0;
                if(--index < 0) {
                    break;
                }
            }
        }

        return C;
    }

    template <std::floating_point T>
    const T& Tensor<T>::operator[](const std::vector<int>& indecies) const {
        int index {m_offset};

        for(int i {}; i < indecies; ++i) {
            index += indecies[i] * m_strides[i];
        }

        return index;
    }

    template <std::floating_point T>
    T& Tensor<T>::operator[](const std::vector<int>& indecies) {
        const Tensor& self = *this;
        return const_cast<T&>(self[indecies]);
    }
}

#endif