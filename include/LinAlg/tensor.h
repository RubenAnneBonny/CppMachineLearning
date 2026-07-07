#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <Rand/random.h>
#include <string>
#include <stdexcept>
#include <concepts>
#include <cstddef>

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
                return static_cast<int>(m_shape.size());
            }

            int num_elements() const {
                int elements {1};

                for(int i {}; i < get_rank(); ++i) {
                    elements *= m_shape[i];
                }

                return elements;
            }

            void calculate_strides() {
                m_strides.clear();
                m_strides.assign(get_rank(), 1);

                for(int j {get_rank() - 2}; j >= 0; --j) {
                    m_strides[j] = m_strides[j + 1] * m_shape[j + 1];
                }
            }

            static bool next_index(std::vector<int>& indecies, const std::vector<int>& shape) {
                for(int i {static_cast<int>(shape.size()) - 1}; i >= 0; --i) {
                    if(++indecies[i] < shape[i]) {
                        return true;
                    }
                    indecies[i] = 0;
                }

                return false;
            }

            static void batching(Tensor& A_view, Tensor& B_view, int not_matching = 0) {
                int A_rank {A_view.get_rank()};
                int B_rank {B_view.get_rank()};
                
                Tensor<T>& max_ref {A_rank > B_rank ? A_view : B_view};
                Tensor<T>& min_ref {A_rank > B_rank ? B_view : A_view};

                int max_rank {max_ref.get_rank()};
                int min_rank {min_ref.get_rank()};

                for(int i {min_rank - 1 - not_matching}, j {max_rank - 1 - not_matching}; i >= 0; --i, --j) {
                    if(max_ref.m_shape[j] != min_ref.m_shape[i]){
                        throw std::invalid_argument(
                            "Cannot perform a batching on two tensors of shape" + 
                            static_cast<std::string>(A_view) + 
                            " and " + 
                            static_cast<std::string>(B_view) + 
                            " the extents of their existing batch axises must match"
                        );
                    }
                }

                min_ref.m_strides.insert(min_ref.m_strides.begin(), max_rank - min_rank, 0);
                min_ref.m_shape.insert(min_ref.m_shape.begin(),
                                       max_ref.m_shape.begin(),
                                       max_ref.m_shape.begin() + max_rank - min_rank);
            }

        public:
            Tensor(const std::vector<int>& shape, T init = 0);

            void normal(Rand::Random<T>& random, T mean, T stddev);

            void uniform(Rand::Random<T>& random, T low, T high);
            
            Tensor copy() const;

            Tensor row(int i) const;

            void unsqueeze(int axis = 0);

            void squeeze(int axis = 0);

            Tensor t() const;

            template <typename Fn>
            void elementwise(Fn fn);

            template <std::floating_point U, typename Fn>
            friend Tensor<U> pairwise(const Tensor<U>& A, const Tensor<U>& B, Fn fn);

            operator std::string() const;

            const T& operator[](const std::vector<int>& indecies) const;
            T& operator[](const std::vector<int>& indecies);

            friend Tensor operator*(const Tensor& A, const Tensor& B) {
                Tensor<T> A_view {A};
                Tensor<T> B_view {B};

                int A_rank {A.get_rank()};
                int B_rank {B.get_rank()};

                int max_rank {A_rank > B_rank ? A_rank : B_rank};
                int min_rank {A_rank > B_rank ? B_rank : A_rank};

                if(min_rank < 2){
                    throw std::invalid_argument(
                        "Cannot perform matrix multiplication on Tensors of shape " + 
                        static_cast<std::string>(A) + 
                        " and " + 
                        static_cast<std::string>(B) +
                        " since one of them has rank < 2"
                    );
                }

                if(A_view.m_shape[A_rank - 1] != B_view.m_shape[B_rank - 2]){
                    throw std::invalid_argument(
                        "Cannot perform matrix multiplication on Tensors of shape " + 
                        static_cast<std::string>(A) + 
                        " and " + 
                        static_cast<std::string>(B) + 
                        " the extent of the last axis on A must match the extent of the next to last axis on B"
                    );
                }

                batching(A_view, B_view, 2);

                std::vector<int> shape {A_view.m_shape};
                shape[max_rank - 1] = B_view.m_shape.back();
                Tensor<T> C {shape};

                std::vector<int> indecies(max_rank, 0);
                std::vector<int> A_indecies(max_rank, 0);
                std::vector<int> B_indecies(max_rank, 0);

                do {
                    for(int i {}; i < max_rank; ++i){
                        A_indecies[i] = indecies[i];
                        B_indecies[i] = indecies[i];
                    }

                    T sum {};
                    for(int i {}; i < A_view.m_shape.back(); ++i){
                        A_indecies[max_rank - 1] = i;
                        B_indecies[max_rank - 2] = i;

                        sum += A_view[A_indecies] * B_view[B_indecies];
                    }

                    C[indecies] = sum;
                } while(next_index(indecies, C.m_shape));

                return C;
            }
            Tensor& operator*=(const Tensor& A) {
                (*this) = (*this) * A;
                return *this;
            }

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
        , m_strides {}
    {
        calculate_strides();
    }

    template <std::floating_point T>
    void Tensor<T>::normal(Rand::Random<T>& random, T mean, T stddev) {
        std::vector<int> indecies(get_rank(), 0);
        do {
            (*this)[indecies] = random.normal(mean, stddev);
        } while(next_index(indecies, m_shape));
    }

    template <std::floating_point T>
    void Tensor<T>::uniform(Rand::Random<T>& random, T low, T high) {
        std::vector<int> indecies(get_rank(), 0);
        do {
            (*this)[indecies] = random.uniform(low, high);
        } while(next_index(indecies, m_shape));
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::copy() const {
        Tensor<T> A {m_shape};

        std::vector<int> indecies(get_rank(), 0);
        do {
            A[indecies] = (*this)[indecies];
        } while(next_index(indecies, m_shape));

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
        int strides {num_elements()};

        if(axis > get_rank() || axis < 0){
            throw std::invalid_argument(
                "Cannot unsqueeze Tensor of shape " +
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since its outside the tensors rank"
            );
        }

        if(axis > 0) {
            strides = m_strides[axis - 1]; 
        }

        m_shape.insert(m_shape.begin() + axis, 1);
        m_strides.insert(m_strides.begin() + axis, strides);
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

        int temp_shape {A.m_shape[rank - 1]};
        A.m_shape[rank - 1] = A.m_shape[rank - 2];
        A.m_shape[rank - 2] = temp_shape;

        int temp_stride {A.m_strides[rank - 1]};
        A.m_strides[rank - 1] = A.m_strides[rank - 2];
        A.m_strides[rank - 2] = temp_stride;

        return A;
    }

    template <std::floating_point T>
    template <typename Fn>
    void Tensor<T>::elementwise(Fn fn) {
        std::vector<int> indecies(get_rank(), 0);
        do {
            (*this)[indecies] = fn((*this)[indecies]);
        } while(next_index(indecies, m_shape));
    }

    template <std::floating_point T, typename Fn>
    Tensor<T> pairwise(const Tensor<T>& A, const Tensor<T>& B, Fn fn) {
        Tensor<T> A_view {A};
        Tensor<T> B_view {B};

        int A_rank {A.get_rank()};
        int B_rank {B.get_rank()};

        int rank {A_rank > B_rank ? A_rank : B_rank};

        Tensor<T>::batching(A_view, B_view);

        Tensor<T> C {A_view.m_shape};

        std::vector<int> indecies(rank, 0);

        do {
            C[indecies] = fn(A_view[indecies], B_view[indecies]);
        } while(next_index(indecies, C.m_shape));

        return C;
    }

    template <std::floating_point T>
    Tensor<T>::operator std::string() const {
        std::string shape_string {"("};

        for(int i {}; i < get_rank(); ++i) {
            shape_string += std::to_string(m_shape[i]);

            if(i < get_rank() - 1) {
                shape_string += ", ";
            } 
        }

        shape_string += ")";

        return shape_string;
    }

    template <std::floating_point T>
    const T& Tensor<T>::operator[](const std::vector<int>& indecies) const {
        int index {m_offset};

        for(int i {}; i < get_rank(); ++i) {
            index += indecies[i] * m_strides[i];
        }

        return (*m_storage)[index];
    }
    template <std::floating_point T>
    T& Tensor<T>::operator[](const std::vector<int>& indecies) {
        const Tensor& self = *this;
        return const_cast<T&>(self[indecies]);
    }
}

#endif