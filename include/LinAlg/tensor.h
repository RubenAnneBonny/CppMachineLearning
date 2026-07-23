#ifndef TENSOR_H
#define TENSOR_H

/*
    Copy construction and copy assignment produces a view sharing storage, copy() is deep copy
    
    These methods return views: {row, slice, unsqueeze, squeeze, t}
    These methods return fresh storage: {copy, gather, pairwise, operator+ - *}
*/

#include <memory>
#include <vector>
#include <Rand/random.h>
#include <string>
#include <stdexcept>
#include <concepts>
#include <cstddef>
#include <cmath>
#include <ostream>

namespace LinAlg {
    template <std::floating_point T>
    class Tensor;

    template <std::floating_point T>
    Tensor<T> one_hot(int extent, int index);

    template <std::floating_point T>
    bool all_close(const Tensor<T>& A, const Tensor<T>& B, T atol, T rtol);

    template <std::floating_point T, std::invocable<T, T> Fn>
    Tensor<T> pairwise(const Tensor<T>& A, const Tensor<T>& B, Fn fn);

    template <std::floating_point T>
    Tensor<T> pairwise_mult(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    Tensor<T> pairwise_add(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    Tensor<T> pairwise_sub(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B);

    template <std::floating_point T>
    class Tensor {
        private:
            std::vector<int> m_shape;
            std::shared_ptr<std::vector<T>> m_storage;
            int m_offset;
            std::vector<int> m_strides;

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
                std::string a_str {static_cast<std::string>(A_view)};
                std::string b_str {static_cast<std::string>(B_view)};

                int A_rank {A_view.get_rank()};
                int B_rank {B_view.get_rank()};

                int max_rank {A_rank > B_rank ? A_rank : B_rank};
                int min_rank {A_rank < B_rank ? A_rank : B_rank};

                {
                    Tensor<T>& min_ref {A_rank > B_rank ? B_view : A_view};
                    min_ref.m_shape.insert(min_ref.m_shape.begin(), max_rank - min_rank, 1);
                    min_ref.m_strides.insert(min_ref.m_strides.begin(), max_rank - min_rank, 0);
                }

                for(int axis {}; axis < max_rank - not_matching; ++axis) {
                    int a_extent {A_view.m_shape[axis]};
                    int b_extent {B_view.m_shape[axis]};

                    if(a_extent == b_extent) {
                        continue;
                    }
                    else if(a_extent == 1) {
                        A_view.m_strides[axis] = 0;
                        A_view.m_shape[axis] = b_extent;
                    }
                    else if(b_extent == 1) {
                        B_view.m_strides[axis] = 0;
                        B_view.m_shape[axis] = a_extent;
                    }
                    else {
                        throw std::invalid_argument(
                            "Cannot braodcast tensors of shape " + 
                            a_str + 
                            " and " + 
                            b_str + 
                            " since axis " + 
                            std::to_string(axis) + 
                            " has incompatible extents " + 
                            std::to_string(a_extent) + 
                            " and " + 
                            std::to_string(b_extent)
                        );
                    }
                }
            }

        public:
            /// @brief Constructor
            /// @param shape The shape of the Tensor
            /// @param init The value to initialize all elements to
            /// @throws std::invalid_argument if a extent of a axis is less than 1
            Tensor(const std::vector<int>& shape, T init = 0);

            /// @brief Randomizes all elements ~N(mean, stddev)
            /// @param random An instance of the random class
            /// @param mean The mean of the elements
            /// @param stddev The standard deviation of all elements
            void normal(Rand::Random<T>& random, T mean, T stddev);

            /// @brief Randomizes all elements ~U(low, high)
            /// @param random An instance of the random class
            /// @param low The lower bound of the elements
            /// @param high The upper bound of the elements
            void uniform(Rand::Random<T>& random, T low, T high);

            /// @brief Sets all elements to value
            /// @param value The value to set all elements to
            void set_all_elements(T value);

            /// @brief Calculates the rank of the tensor
            /// @return The rank
            int get_rank() const {
                return static_cast<int>(m_shape.size());
            }

            /// @brief Returns the extent of an axis
            /// @throws std::invalid_argument if axis is outside the rank of the tensor
            int get_extent(int axis) const {
                if(axis < 0 || axis >= get_rank()) {
                    throw std::invalid_argument(
                        "Cannot get extent on axis " + 
                        std::to_string(axis) + 
                        " on tensor " + 
                        static_cast<std::string>(*this) + 
                        " since it doesnt have that axis"
                    );
                }

                return m_shape[static_cast<std::size_t>(axis)];
            }
            
            /// @brief Creates a deep copy of the tensor
            /// @return Returns a copy of the tensor
            Tensor copy() const;

            /// @brief Creates a view of the one of the tensor by removing first axis
            /// @param i The index of the sub-tensor to create a view of
            /// @return The tensor-view
            /// @throws std::invalid_argument if i is outside extent of first axis
            Tensor row(int i) const;

            /// @brief Creates a slice view of the tensor
            /// @param start Where to start the slice of the tensor (inclusive)
            /// @param end Where to end the slice of the tensor (exclusive)
            /// @return A tensor view
            /// @throws std::invalid_argument if start and end isnt a valid range in the extent of first axis
            Tensor slice(int start, int end) const;

            /// @brief Constructs a new tensor, where the rows are the rows corresponding to to_gather
            /// @param to_gather The indecies of the rows to gather
            /// @return A tensor of shape (to_gather size, ...the same shapes as this)
            /// @throws std::invalid_argument if to_gather is empty
            /// @throws std::invalid_argument if any element in to_gather is outside extent of first axis
            Tensor gather(const std::vector<int>& to_gather) const;

            /// @brief Add an extra axis with extent 1
            /// @param axis Before which axis to add the new
            /// @return A tensor-view of the unsqueezed tensor
            /// @throws std::invalid_argument if axis is outside the rank of the tensor
            Tensor unsqueeze(int axis = 0) const;

            /// @brief Removes an axis with extent 1
            /// @param axis Which axis to remove
            /// @return A tensor-view of the squeezed tensor
            /// @throws std::invalid_argument if tensor rank is 1
            /// @throws std::invalid_argument if extent of axis is not 1
            /// @throws std::invalid_argument if axis is outside the rank of the tensor
            Tensor squeeze(int axis = 0) const;

            /// @brief Transposes the last two axises of the tensor
            /// @return A tensor-view of the transposed tensor
            /// @throws std::invalid_argument if rank of tensor is less than 2
            Tensor t() const;

            /// @brief Finds the position of the largest element in the tensor
            /// @return The position of the largest element
            std::vector<int> argmax() const;

            /// @brief Computes the sum over all elements in the tensor
            /// @return The sum
            T sum() const;

            /// @brief Finds the max over all elements in the tensor
            /// @return The maximum value
            T max() const;

            /// @brief Creates a tensor of rank 1 with 1 one and the rest zeros
            /// @param extent The extent of the only axis
            /// @param index The index of the 1
            /// @return The created tensor
            friend Tensor<T> one_hot<T>(int extent, int index);

            /// @brief Performs a function on each element of the tensor
            /// @tparam Fn A type of function that takes only one parameter of type T
            /// @param fn The lambda function to perform on each element
            /// @return A reference to this tensor
            template <std::invocable<T> Fn>
            Tensor& elementwise(Fn fn);

            /// @brief Creates tensor where each element is the result of a function between two tensors, uses batching
            /// @tparam Fn A type of function that takes two parameters of type T
            /// @tparam U The same type as T
            /// @param A The first tensor
            /// @param B The second tensor
            /// @param fn The lambda function to perform on the tensors
            /// @return A new tensor, the result
            /// @throws std::invalid_argument if batching cannot be performed
            template <std::floating_point U, std::invocable<U, U> Fn>
            friend Tensor<U> pairwise(const Tensor<U>& A, const Tensor<U>& B, Fn fn);

            /// @brief Allows static_cast<std::string>(tensor), returns string that displays shape nicely
            operator std::string() const;

            /// @brief Allows printing a tensor using the std::string() operator
            friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& A) {
                return os << static_cast<std::string>(A);
            }

            /// @brief Allows accessing elements in the tensor
            /// @param indecies The indecies of each axis to retrive element at
            /// @return A reference to the element
            const T& operator[](const std::vector<int>& indecies) const;
            T& operator[](const std::vector<int>& indecies);

            /// @brief Checks for equality between two tensors
            /// @param A The first tensor
            /// @param B The second tensor
            /// @return True if equal, false otherwise
            friend bool operator==(const Tensor& A, const Tensor& B) {
                if(A.get_rank() != B.get_rank()) {
                    return false;
                }

                for(int i {}; i < A.get_rank(); ++i) {
                    if(A.m_shape[i] != B.m_shape[i]){
                        return false;
                    }
                }

                std::vector<int> indecies(A.get_rank(), 0);

                do {
                    if(A[indecies] != B[indecies]) {
                        return false;
                    }
                } while(next_index(indecies, A.m_shape));

                return true;
            }
            friend bool operator!=(const Tensor& A, const Tensor& B) {
                return !(A == B);
            }

            /// @brief Similar to operator==, but uses tolerance so one can compare floating point types
            /// @param A The first tensor
            /// @param B The second tensor 
            /// @param atol The absolute tolerance
            /// @param rtol The relative tolerance
            /// @return True if A and B are within the tolerance from each other, false otherwise
            friend bool all_close<T>(const Tensor<T>& A, const Tensor<T>& B, T atol, T rtol);

            /// @brief Performs pairwise addition
            /// @param A The first tensor
            /// @param B The second tensor
            /// @return A new tensor, the result of the operation
            /// @throws std::invalid_argument if batching cannot be performed
            friend Tensor operator+(const Tensor& A, const Tensor& B) {
                auto addition{
                [](T a, T b)
                {
                    return a + b;
                }
                };

                return pairwise(A, B, addition);
            }
            friend Tensor<T> pairwise_add<T>(const Tensor<T>& A, const Tensor<T>& B);
            Tensor& operator+=(const Tensor<T>& A) {
                auto addition{
                [](T a, T b)
                {
                    return a + b;
                }
                };

                Tensor<T> C {pairwise(*this, A, addition)};

                if(C.m_shape != m_shape) {
                    throw std::invalid_argument(
                        "Cannot perform += between tensors of shape " + 
                        static_cast<std::string>(*this) + 
                        " and " + 
                        static_cast<std::string>(A) + 
                        " since it would change its shape"
                    );
                }

                std::vector<int> indecies(get_rank(), 0);
                do {
                    (*this)[indecies] = C[indecies];
                } while(next_index(indecies, m_shape));

                return *this;
            }

            friend Tensor operator+(const Tensor& A, T b) {
                auto addition{
                    [b](T a)
                    {
                        return a + b;
                    }
                };

                Tensor<T> C {A.copy()};

                return C.elementwise(addition);
            }
            friend Tensor operator+(T b, const Tensor& A) {
                return A + b;
            }
            Tensor& operator+=(T a) {
                auto addition{
                    [a](T b)
                    {
                        return a + b;
                    }
                };

                return elementwise(addition);
            }

            /// @brief Performs pairwise subtraction
            /// @param A The first tensor
            /// @param B The second tensor
            /// @return A new tensor, the result of the operation
            /// @throws std::invalid_argument if batching cannot be performed
            friend Tensor operator-(const Tensor& A, const Tensor& B) {
                auto subtraction{
                [](T a, T b)
                {
                    return a - b;
                }
                };

                return pairwise(A, B, subtraction);
            }
            friend Tensor<T> pairwise_sub<T>(const Tensor<T>& A, const Tensor<T>& B);
            Tensor& operator-=(const Tensor<T>& A) {
                auto subtraction{
                [](T a, T b)
                {
                    return a - b;
                }
                };

                Tensor<T> C {pairwise(*this, A, subtraction)};

                if(C.m_shape != m_shape) {
                    throw std::invalid_argument(
                        "Cannot perform -= between tensors of shape " + 
                        static_cast<std::string>(*this) + 
                        " and " + 
                        static_cast<std::string>(A) + 
                        " since it would change its shape"
                    );
                }

                std::vector<int> indecies(get_rank(), 0);
                do {
                    (*this)[indecies] = C[indecies];
                } while(next_index(indecies, m_shape));

                return *this;
            }

            Tensor operator-() const {
                return (*this) * (-1);
            }

            friend Tensor operator-(const Tensor& A, T b) {
                auto subtraction{
                    [b](T a)
                    {
                        return a - b;
                    }
                };

                Tensor<T> C {A.copy()};

                return C.elementwise(subtraction);
            }
            friend Tensor operator-(T b, const Tensor& A) {
                auto subtraction{
                    [b](T a)
                    {
                        return b - a;
                    }
                };

                Tensor<T> C {A.copy()};

                return C.elementwise(subtraction);
            }
            Tensor& operator-=(T a) {
                auto subtraction{
                    [a](T b)
                    {
                        return b - a;
                    }
                };

                return elementwise(subtraction);
            }

            /// @brief Performs pairwise multiplication
            /// @param A The first tensor
            /// @param B The second tensor
            /// @return A new tensor the result of the operation
            /// @throws std::invalid_argument if batching cannot be 
            friend Tensor operator*(const Tensor& A, const Tensor& B) {
                auto multiplication{
                    [](T a, T b)
                    {
                        return a * b;
                    }
                };

                return pairwise(A, B, multiplication);
            }
            Tensor& operator*=(const Tensor& A) {
                Tensor<T> C {(*this) * A};

                if(C.m_shape != m_shape) {
                    throw std::invalid_argument(
                        "Cannot perform *= between tensors of shape " + 
                        static_cast<std::string>(*this) + 
                        " and " + 
                        static_cast<std::string>(A) + 
                        " since it would change its shape"
                    );
                }

                std::vector<int> indecies(get_rank(), 0);
                do {
                    (*this)[{indecies}] = C[indecies];
                } while(next_index(indecies, m_shape));

                return *this;
            }
            friend Tensor<T> pairwise_mult<T>(const Tensor<T>& A, const Tensor<T>& B);

            friend Tensor operator*(const Tensor& A, T b) {
                auto multiplication{
                    [b](T a)
                    {
                        return a * b;
                    }
                };

                Tensor<T> C {A.copy()};

                return C.elementwise(multiplication);
            }
            friend Tensor operator*(T b, const Tensor& A) {
                return A * b;
            }
            Tensor& operator*=(T a) {
                auto multiplication{
                    [a](T b)
                    {
                        return a * b;
                    }
                };

                return elementwise(multiplication);
            }
               
            /// @brief Performs matrix multiplication, uses batching
            /// @param A The first tensor
            /// @param B The second tensor
            /// @return The result of the operation
            /// @throws std::invalid_argument if batching cannot be performed
            /// @throws std::invalid_argument if the rank of either A or B is less than 2
            /// @throws std::invalid_argument if the extent of the last axis of A dont match the extent of next to last axis of B
            friend Tensor<T> matmul<T>(const Tensor<T>& A, const Tensor<T>& B);
    };

    template <std::floating_point T>
    Tensor<T>::Tensor(const std::vector<int>& shape, T init)
        : m_shape {shape}
        , m_storage {std::make_shared<std::vector<T>>(num_elements(), init)}
        , m_offset {}
        , m_strides {}
    {
        for(int i {}; i < static_cast<int>(shape.size()); ++i) {
            if(shape[i] < 1) {
                throw std::invalid_argument(
                    "Cannot create tensor with any extent less than 1"
                );
            }
        }

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
    void Tensor<T>::set_all_elements(T value) {
        std::vector<int> indecies(get_rank(), 0);
        do {
            (*this)[indecies] = value;
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
                "Cannot use .row on tensor with rank 1"
            );
        }

        if(i < 0 || i >= m_shape[0]) {
            throw std::invalid_argument(
                "Cannot get row " + 
                std::to_string(i) + 
                " of tensor " + 
                static_cast<std::string>(*this) + 
                " since " + 
                std::to_string(i) + 
                " is outside the extent of the first axis"
            );
        }

        Tensor A {*this};
        A.m_offset += i * m_strides[0];
        A.m_shape.erase(A.m_shape.begin());
        A.m_strides.erase(A.m_strides.begin());

        return A;
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::slice(int start, int end) const {
        Tensor<T> A {*this};

        if(start < 0 || end > get_extent(0) || end <= start) {
            throw std::invalid_argument(
                "Cannot slice Tensor of shape " + 
                static_cast<std::string>(*this) + 
                " since the indecies for start (" + 
                std::to_string(start) + 
                ") and end (" + 
                std::to_string(end) + 
                ") must lie inside the extent of the first axis"
            );
        }

        A.m_offset += start * m_strides[0];
        A.m_shape[0] = end - start;

        return A;
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::gather(const std::vector<int>& to_gather) const {
        if(to_gather.empty()) {
            throw std::invalid_argument(
                "Cannot gather a tensor with and empty input vector"
            );
        } 

        for(int element : to_gather) {
            if(element < 0 || element >= get_extent(0)) {
                throw std::invalid_argument(
                    "Cannot gather Tensor according to input vector since it contains " + 
                    std::to_string(element) + 
                    " which it lies out of the extent of the first axis for tensor of shape " + 
                    static_cast<std::string>(*this)
                );
            }
        }

        std::vector<int> shape {m_shape};
        shape[0] = static_cast<int>(to_gather.size());
        Tensor<T> A {shape};

        std::vector<int> indecies(A.get_rank(), 0);
        std::vector<int> this_indecies(A.get_rank());

        do {
            this_indecies = indecies;
            this_indecies[0] = to_gather[indecies[0]];

            A[indecies] = (*this)[this_indecies];
        } while(next_index(indecies, A.m_shape));

        return A;
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::unsqueeze(int axis) const {
        Tensor<T> A {*this};

        if(axis > A.get_rank() || axis < 0){
            throw std::invalid_argument(
                "Cannot unsqueeze Tensor of shape " +
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since its outside the tensors rank"
            );
        }

        int strides {axis == 0 ? m_strides[0] * m_shape[0] : m_strides[axis - 1]};

        A.m_shape.insert(A.m_shape.begin() + axis, 1);
        A.m_strides.insert(A.m_strides.begin() + axis, strides);

        return A;
    }

    template <std::floating_point T>
    Tensor<T> Tensor<T>::squeeze(int axis) const {
        if(get_rank() == 1) {
            throw std::invalid_argument(
                "Cannot squeeze tensor of shape " + 
                static_cast<std::string>(*this) + 
                " since it would result in a rank of zero"
            );
        }

        if(axis >= get_rank() || axis < 0){
            throw std::invalid_argument(
                "Cannot squeeze tensor of shape " +
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since its outisde its rank"
            );
        }

        if(m_shape[axis] != 1){
            throw std::invalid_argument(
                "Cannot squeeze tensor of shape " + 
                static_cast<std::string>(*this) + 
                " in axis " + 
                std::to_string(axis) + 
                " since the extent of that axis is not 1"
            );
        }

        Tensor<T> A {*this};

        A.m_shape.erase(A.m_shape.begin() + axis);
        A.m_strides.erase(A.m_strides.begin() + axis);

        return A;
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
    std::vector<int> Tensor<T>::argmax() const {
        std::vector<int> best_indecies(get_rank(), 0);
        T max_element {(*this)[best_indecies]};

        std::vector<int> indecies(get_rank(), 0);

        do {
            T element = (*this)[indecies];

            if(element > max_element) {
                max_element = element;
                best_indecies = indecies;
            }
        } while(next_index(indecies, m_shape));

        return best_indecies;
    }

    template <std::floating_point T>
    T Tensor<T>::sum() const {
        std::vector<int> indecies(get_rank(), 0);

        T sum {};

        do {
            sum += (*this)[indecies];
        } while(next_index(indecies, m_shape));

        return sum;
    }

    template <std::floating_point T>
    T Tensor<T>::max() const {
        std::vector<int> indecies(get_rank(), 0);

        T max_value = (*this)[indecies];

        do {
            T element = (*this)[indecies];

            max_value = (element > max_value ? element : max_value);
        } while(next_index(indecies, m_shape));

        return max_value;
    }

    template <std::floating_point T>
    Tensor<T> one_hot(int extent, int index) {
        LinAlg::Tensor<T> X {{extent}};

        X[{index}] = 1;

        return X;
    }

    template <std::floating_point T>
    template <std::invocable<T> Fn>
    Tensor<T>& Tensor<T>::elementwise(Fn fn) {
        std::vector<int> indecies(get_rank(), 0);

        do {
            (*this)[indecies] = fn((*this)[indecies]);
        } while(next_index(indecies, m_shape));

        return *this;
    }

    template <std::floating_point T, std::invocable<T, T> Fn>
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
        } while(Tensor<T>::next_index(indecies, C.m_shape));

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

        return (*m_storage)[static_cast<std::size_t>(index)];
    }
    template <std::floating_point T>
    T& Tensor<T>::operator[](const std::vector<int>& indecies) {
        const Tensor& self = *this;
        return const_cast<T&>(self[indecies]);
    }

    template <std::floating_point T>
    bool all_close(const Tensor<T>& A, const Tensor<T>& B, T atol, T rtol) {
        if(A.get_rank() != B.get_rank()) {
            return false;
        }

        for(int i {}; i < A.get_rank(); ++i) {
            if(A.m_shape[i] != B.m_shape[i]){
                return false;
            }
        }

        std::vector<int> indecies(A.get_rank(), 0);

        do {
            if(std::abs(A[indecies] - B[indecies]) > atol + std::abs(B[indecies]) * rtol) {
                return false;
            }
        } while(Tensor<T>::next_index(indecies, A.m_shape));

        return true;
    }

    template <std::floating_point T>
    Tensor<T> pairwise_add(const Tensor<T>& A, const Tensor<T>& B) {
        return A + B;
    }

    template <std::floating_point T>
    Tensor<T> pairwise_sub(const Tensor<T>& A, const Tensor<T>& B) {
        return A - B;
    }

    template <std::floating_point T>
    Tensor<T> pairwise_mult(const Tensor<T>& A, const Tensor<T>& B) {
        return A * B;
    }

    template <std::floating_point T>
    Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
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

        Tensor<T>::batching(A_view, B_view, 2);

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
        } while(Tensor<T>::next_index(indecies, C.m_shape));

        return C;
    }
}

#endif