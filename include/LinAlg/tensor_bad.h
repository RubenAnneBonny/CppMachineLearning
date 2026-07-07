#ifndef TENSOR_BAD_H
#define TENSOR_BAD_H

#include <vector>
#include <string>
#include <stdexcept>
#include <Rand/random.h>
#include <concepts>

namespace LinAlg {
    template <typename T>
    class Tensor;

    template <typename T>
    Tensor<T> pairwise_mult(const Tensor<T>& A, const Tensor<T>& B);

    template <typename T>
    Tensor<T> pairwise_add(const Tensor<T>& A, const Tensor<T>& B);

    template <typename T>
    class Tensor{
        private:
            std::vector<int> m_dim_list;
            std::vector<int> m_suffix_values;
            std::vector<T> m_values;

            /// @brief Calculate the m_suffix_values
            /// @return The total amount of elements in the tensor
            int calculate_suffix_values();

            int num_elements() const {
                return static_cast<int>(m_values.size());
            }

            /// @brief Used for safety checking, can we broadcast either A or B to be same dimensions
            static bool same_dimensions_with_broadcasting(const Tensor& A, const Tensor& B) {
                int A_dim {A.get_dim()};
                int B_dim {B.get_dim()};

                int max_dim {A_dim > B_dim ? A_dim : B_dim};
                int min_dim {A_dim > B_dim ? B_dim : A_dim};

                for(int i {}; i < min_dim; ++i){
                    if(A.get_dim_at_index(i + A_dim - min_dim) != B.get_dim_at_index(i + B_dim - min_dim)){
                        return false;
                    }
                }

                return true;
            }

            /// @brief Does pariwise fn(A, B) using broadcasting
            template <typename Fn>
            static Tensor pairwise_operation(const Tensor& A, const Tensor& B, Fn fn){
                if(!same_dimensions_with_broadcasting(A, B)){
                    throw std::invalid_argument(
                        "Cannot do pairwise_operation on non-broadcast compatible tensors"
                    );
                }

                int A_dim {A.get_dim()};
                int B_dim {B.get_dim()};

                {
                    const Tensor<T>& ref_max {A_dim >= B_dim ? A : B};
                    const Tensor<T>& ref_min {A_dim < B_dim ? A : B};

                    Tensor<T> C {ref_max.m_dim_list};

                    int min_ind {};
                    for(int i {}; i < ref_max.num_elements(); ++i){
                        C.m_values[i] = fn(ref_max.m_values[i], ref_min.m_values[min_ind]);

                        if(++min_ind >= ref_min.num_elements()){
                            min_ind = 0;
                        }
                    }

                    return C;
                }
            }

        public:
            /// @brief Constructor, initializer-initializes m_values, and calculate m_suffix_values
            /// @param dim_list A list of all the sizes for all dimensions 
            /// @param initializer The value to initialize all tensor values
            Tensor(const std::vector<int>& dim_list, T initializer = 0);

            /// @brief Constructor for creating a new tensor from several others
            /// @param tensors The tensors to merge
            /// @param unsqueeze The dimension to merge them in
            /// @warning Unlike the rest of this class, this throws no warnings for invalid input (this needs to be fast)
            explicit Tensor(const std::vector<Tensor<T>>& tensors, int unsqueeze);

            /// @brief Sets elements randomly ~N(mean, stddev)
            template <std::floating_point U = T>
                requires std::same_as<U, T>
            void normal(Rand::Random<U>& random, T mean, T stddev);

            /// @brief Sets elements randomly ~U(low, high)
            template <std::floating_point U = T>
                requires std::same_as<U, T>
            void uniform(Rand::Random<U>& random, T low, T high);

            std::vector<int> get_dim_list() const {
                return m_dim_list;
            }

            int get_dim() const {
                return static_cast<int>(m_dim_list.size());
            }

            int get_dim_at_index(int index) const {
                if(index < 0 || index >= get_dim()){
                    throw std::invalid_argument(
                        "Attempted indexing in m_dim_list out of bounds"
                    );
                }

                return m_dim_list[index];
            }

            /// @brief Does elementwise fn(A)
            template <typename Fn>
            void elementwise_operation(Fn fn);

            /// @brief Adds extra dimension to specified index in m_dim_list
            /// @param index The position to insert dimension
            void unsqueeze(int index = 0);

            /// @brief Removes dimension of 1
            /// @param index The index of dimension to remove
            void squeeze(int index = 0);

            /// @brief Transposes the last 2 dimensions of tensor
            Tensor t() const;

            /// @brief Allows static_cast from tensor to string, displays the m_dim_list nicely
            operator std::string() const;

            /// @brief Access operator for the elements in the tensor
            /// @param indecies A vector of the indecies for the element being searched for
            /// @return The element at the indecies position
            const T& operator()(const std::vector<int>& indecies) const;
            T& operator()(const std::vector<int>& indecies);

            /// @brief Same as operator() but doesnt throw warnings
            const T& unsafe_access(const std::vector<int>& indecies) const;
            T& unsafe_access(const std::vector<int>& indecies);

            /// @brief Matrix multiplication between two innermost dimensions, for each of the batches
            /// @return The result of the operation
            friend Tensor operator*(const Tensor& A, const Tensor& B){
                int A_dim {A.get_dim()};
                int B_dim {B.get_dim()};

                int max_dim {A_dim > B_dim ? A_dim : B_dim};
                int min_dim {A_dim > B_dim ? B_dim : A_dim};

                if(min_dim <= 1){
                    throw std::invalid_argument(
                        "Tensor must be atleast of dimension 2 to multiply"
                    );
                }

                for(int i {}; i < min_dim - 2; ++i){
                    if(A.get_dim_at_index(i + A_dim - min_dim) != B.get_dim_at_index(i + B_dim - min_dim)){
                        throw std::invalid_argument(
                            "For tensor multiplication dimensions above innermost 2, must be equal size"
                        );
                    }
                }

                int A_rows {A.get_dim_at_index(A_dim - 2)};
                int A_cols {A.get_dim_at_index(A_dim - 1)};
                int B_rows {B.get_dim_at_index(B_dim - 2)};
                int B_cols {B.get_dim_at_index(B_dim - 1)};

                if(A_cols != B_rows){
                    throw std::invalid_argument(
                        "For tensor multiplication the last dimension of A must match the next to last dimension of B"
                    );
                }

                int num_batches {1};
                std::vector<int> dimensions(max_dim);
                {
                    const Tensor<T>& ref_max {A_dim == max_dim ? A : B};

                    for(int i {}; i < max_dim - 2; ++i){
                        dimensions[i] = ref_max.get_dim_at_index(i);
                        num_batches *= ref_max.get_dim_at_index(i);
                    }
                }

                dimensions[max_dim - 2] = A_rows;
                dimensions[max_dim - 1] = B_cols;

                Tensor C {dimensions};    

                int A_matrix_size {A_rows * A_cols};
                int B_matrix_size {B_rows * B_cols};
                int C_matrix_size {A_rows * B_cols};
                int A_batch_size {};
                int B_batch_size {};
                int C_batch_size {};

                for(int b {}; b < num_batches; ++b){
                    for(int i {}; i < A_rows; ++i){
                        for(int j {}; j < A_cols; ++j){
                            T a = A.m_values[A_batch_size + i * A_cols + j];

                            for(int k {}; k < B_cols; ++k){
                                C.m_values[C_batch_size + i * B_cols + k] += a * B.m_values[B_batch_size + j * B_cols + k];
                            }
                        }
                    }

                    A_batch_size += A_matrix_size;
                    B_batch_size += B_matrix_size;
                    C_batch_size += C_matrix_size;

                    if(A_batch_size >= static_cast<int>(A.m_values.size())){
                        A_batch_size = 0;
                    }
                    if(B_batch_size >= static_cast<int>(B.m_values.size())){
                        B_batch_size = 0;
                    }
                }

                return C;
            }

            Tensor& operator*=(const Tensor& A){
                (*this) = (*this) * A;
                return *this;
            }

            /// @brief Compares two tensors, returns true, if there dimensions and elements match, false otherwise
            /// @return true if their dimensions and elements match, false otherwise
            friend bool operator==(const Tensor& A, const Tensor& B){
                if(A.get_dim() != B.get_dim()) return false;
                for(int i {}; i < A.get_dim(); ++i){
                    if(A.m_dim_list[i] != B.m_dim_list[i]){
                        return false;
                    }
                }

                for(int i {}; i < static_cast<int>(A.m_values.size()); ++i){
                    if(A.m_values[i] != B.m_values[i]){
                        return false;
                    }
                }

                return true;
            }

            /// @brief The opposite of ==
            friend bool operator!=(const Tensor& A, const Tensor& B){
                return !(A == B);
            }

            /// @brief Does pairwise addition with broadcasting
            /// @return The result of the operation
            friend Tensor operator+(const Tensor& A, const Tensor& B){
                auto addition{
                    [](T a, T b)
                    {
                        return a + b;
                    }
                };

                return Tensor<T>::pairwise_operation(A, B, addition);
            }
            friend Tensor<T> pairwise_add<T>(const Tensor<T>& A, const Tensor<T>& B);

            /// @brief Does elementwise addition to A by b
            /// @return The result of the operation
            friend Tensor operator+(const Tensor& A, T b){
                auto addition{
                    [b](T a)
                    {
                        return a + b;
                    }
                };

                Tensor<T> B {A};

                B.elementwise_operation(addition);

                return B;
            }
            friend Tensor operator+(T b, const Tensor& A){
                return A + b;
            }
            Tensor& operator+=(T a){
                (*this) = (*this) + a;
                return this;
            }
            
            /// @brief Does pairwise multiplication with broadcasting
            /// @return The result of the operation
            friend Tensor<T> pairwise_mult<T>(const Tensor<T>& A, const Tensor<T>& B);

            /// @brief Does elementwise multipliction to A by b
            /// @return The result of the operation
            friend Tensor operator*(const Tensor& A, T b){
                auto multiplication{
                    [b](T a)
                    {
                        return a * b;
                    }
                };

                Tensor<T> B {A};

                B.elementwise_operation(multiplication);

                return B;
            }
            friend Tensor operator*(T b, const Tensor& A){
                return A * b;
            }
            Tensor& operator*=(T b){
                (*this) = (*this) * b;
                return this;
            }
    };

    template <typename T>
    int Tensor<T>::calculate_suffix_values() {
        m_suffix_values.clear();
        m_suffix_values.assign(get_dim(), 0);

        int length {1};
        for(int i {get_dim() - 1}; i >= 0; --i){
            m_suffix_values[i] = length;
            length *= m_dim_list[i];
        }

        return length;
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<int>& dim_list, T initializer)
        : m_dim_list{dim_list}
    {
        int length {calculate_suffix_values()};

        m_values.assign(length, initializer);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<Tensor>& tensors, int unsqueeze)
    {
        std::vector<int> dimensions {tensors[0].m_dim_list};
        int size {static_cast<int>(tensors.size())};

        for(int i {1}; i < size; ++i){
            dimensions[unsqueeze] += tensors[i].m_dim_list[unsqueeze];
        }

        Tensor A {dimensions};
        int index {};
        int inner_index {};

        for(int i {}; i < A.num_elements(); ++i){
            A.m_values[i] = tensors[index].m_values[inner_index];

            if(++inner_index >= tensors[index].num_elements()){
                inner_index = 0;
                ++index;
            }
        }

        *this = A;
    }

    template <typename T>
    template <typename Fn>
    void Tensor<T>::elementwise_operation(Fn fn){
        for(int i {}; i < num_elements(); ++i){
            m_values[i] = fn(m_values[i]);
        }
    }

    template <typename T>
    template <std::floating_point U>
        requires std::same_as<U, T>
    void Tensor<T>::normal(Rand::Random<U>& random, T mean, T stddev) {
        for(int i {}; i < num_elements(); ++i){
            m_values[i] = random.normal(mean, stddev);
        }
    }

    template <typename T>
    template <std::floating_point U>
        requires std::same_as<U, T>
    void Tensor<T>::uniform(Rand::Random<U>& random, T low, T high) {
        for(int i {}; i < num_elements(); ++i){
            m_values[i] = random.uniform(low, high);
        }
    }

    template <typename T>
    void Tensor<T>::unsqueeze(int index) {
        if(index < 0 || index > get_dim()){
            throw std::invalid_argument(
                "Cannot unsqueeze a tensor outside of its dimensions"
            );
        }

        m_dim_list.insert(m_dim_list.begin() + index, 1);

        calculate_suffix_values();
    }

    template <typename T>
    void Tensor<T>::squeeze(int index) {
        if(index < 0 || index >= get_dim()){
            throw std::invalid_argument(
                "Cannot squeeze dimension that tensor do not have"
            );
        }

        if(m_dim_list[index] != 1){
            throw std::invalid_argument(
                "Cannot squeeze dimension with other value than 1"
            );
        }

        m_dim_list.erase(m_dim_list.begin() + index);

        calculate_suffix_values();
    }

    template <typename T>
    Tensor<T> Tensor<T>::t() const {
        int num_batches {1};
        int dim {get_dim()};

        if(dim < 2){
            throw std::invalid_argument(
                "Cannot transpose tensor with dimensions < 2"
            );
        }

        int rows {get_dim_at_index(dim - 2)};
        int cols {get_dim_at_index(dim - 1)};

        for(int i {}; i < dim - 2; ++i){
            num_batches *= get_dim_at_index(i);
        }

        int matrix_size {rows * cols};
        int batch_size {};

        std::vector<int> dimensions(dim);
        for(int i {}; i < dim - 2; ++i){
            dimensions[i] = m_dim_list[i];
        }

        dimensions[dim - 2] = m_dim_list[dim - 1];
        dimensions[dim - 1] = m_dim_list[dim - 2];

        Tensor<T> t {dimensions};

        for(int b {}; b < num_batches; ++b){
            for(int i {}; i < rows; ++i){
                for(int j {}; j < cols; ++j){
                    t.m_values[batch_size + j * rows + i] = (*this).m_values[batch_size + i * cols + j];
                }
            }

            batch_size += matrix_size;
        }

        return t;
    }

    template <typename T>
    Tensor<T>::operator std::string() const{
        std::string ans {"("};
        for(int i = 0; i < get_dim(); ++i){
            ans += std::to_string(m_dim_list[i]);
            if(i != get_dim() - 1){
                ans += ", ";
            }
        }
        ans += ")";

        return ans;
    }

    template <typename T>
    const T& Tensor<T>::unsafe_access(const std::vector<int>& indecies) const{
        int ind {};
        for(int i {}; i < get_dim(); ++i){
            ind += indecies[i] * m_suffix_values[i]; 
        }

        return m_values[ind];
    }

    template <typename T>
    T& Tensor<T>::unsafe_access(const std::vector<int>& indecies){
        const Tensor& self = *this;  
        return const_cast<T&>(self(indecies));
    }

    template <typename T>
    const T& Tensor<T>::operator()(const std::vector<int>& indecies) const{
        if(static_cast<int>(indecies.size()) != get_dim()){
            throw std::invalid_argument(
                "Index used " + 
                std::to_string(indecies.size()) + 
                " dimensions, while tensor has dimension " +
                static_cast<std::string>(*this)
            );
        }

        bool correct_indecies = true;
        for(int i {}; i < get_dim(); ++i){
            if(indecies[i] < 0 || indecies[i] >= m_dim_list[i]){
                correct_indecies = false;
            }
        }

        if(!correct_indecies){
            throw std::invalid_argument(
                "Index out of bounds!"
            );
        }

        return unsafe_access(indecies);
    }

    template <typename T>
    T& Tensor<T>::operator()(const std::vector<int>& indecies){
        const Tensor& self = *this;  
        return const_cast<T&>(self(indecies));
    }

    template <typename T>
    Tensor<T> pairwise_mult(const Tensor<T>& A, const Tensor<T>& B) {
        auto multiplication{
            [](T a, T b)
            {
                return a * b;
            }
        };

        return Tensor<T>::pairwise_operation(A, B, multiplication);
    }

    template <typename T>
    Tensor<T> pairwise_add(const Tensor<T>& A, const Tensor<T>& B) {
        return A + B;
    }
}

#endif