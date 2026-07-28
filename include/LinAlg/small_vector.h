#ifndef CML_SMALL_VECTOR_H
#define CML_SMALL_VECTOR_H

#include <cstddef>
#include <initializer_list>
#include <algorithm>
#include <utility>
#include <vector>

namespace LinAlg {
    /**
     * @brief A vector that allows for stack-allocation for small vector of size
     * <= N
     *
     * @details When performing operations on this vector that alters its size
     * and it exceeds N, it will allocate memory on the heap. Its size will at
     * least double each time allocating new memory
     * 
     * @code
     * LinAlg::Small_vector<double, 4> v {1, -2, 9};
     * v[1] = 2;
     * @endcode
     */
    template <typename T, std::size_t N>
    class Small_vector {
        private:
            T m_inline[N] {};
            T* m_data {m_inline};
            std::size_t m_size {0};
            std::size_t m_cap {N};

            bool on_heap() const {
                return m_data != m_inline;
            }

            void grow_to(std::size_t needed) {
                if(needed <= m_cap) {
                    return;
                }
                std::size_t new_cap {m_cap * 2 > needed ? m_cap * 2 : needed};
                T* fresh {new T[new_cap]};
                std::copy(m_data, m_data + m_size, fresh);
                if(on_heap()) {
                    delete[] m_data;
                }
                m_data = fresh;
                m_cap = new_cap;
            }
        
        public:
            /// @brief Default constructor for the small vector
            Small_vector() = default;

            /// @brief Constructor for small vector
            /// @param count The amount of elements to allocate memory for
            explicit Small_vector(std::size_t count) { 
                assign(count, T{}); 
            }

            /// @brief Constructor for small vector
            /// @param count The amount of elements to allocate memory for
            /// @param value The value to initialize all elements to
            Small_vector(std::size_t count, const T& value) {
                assign(count, value);
            }

            /// @brief Constructor for small vector
            /// @param v A vector to initialize the small vector with
            Small_vector(const std::vector<T>& v) {
                grow_to(v.size());
                std::copy(v.begin(), v.end(), m_data);
                m_size = v.size();
            }

            /// @brief Constructor for small vector
            /// @param init An intializer_list to initialize the small vector with
            Small_vector(std::initializer_list<T> init) {
                grow_to(init.size());
                std::copy(init.begin(), init.end(), m_data);
                m_size = init.size();
            }

            /// @brief Constructor for small vector
            /// @param other An other small vector to make a copy of
            Small_vector(const Small_vector& other) {
                grow_to(other.m_size);
                std::copy(other.m_data, other.m_data + other.m_size, m_data);
                m_size = other.m_size;
            }

            /// @brief Defines the operator= by copying
            /// @param other The small vector to copy data from
            /// @return A reference to this small vector
            Small_vector& operator=(const Small_vector& other) {
                if(this != &other) {
                    m_size = 0;
                    grow_to(other.m_size);
                    std::copy(other.m_data, other.m_data + other.m_size, m_data);
                    m_size = other.m_size;
                }
                return *this;
            }

            /// @brief Constructor for small vector
            /// @param other An rvalue reference to move data from
            Small_vector(Small_vector&& other) noexcept {
                if(other.on_heap()) {
                    m_data = other.m_data;
                    m_cap = other.m_cap;
                    m_size = other.m_size;
                    other.m_data = other.m_inline;
                    other.m_cap = N;
                    other.m_size = 0;
                }
                else {
                    std::copy(other.m_data, other.m_data + other.m_size, m_inline);
                    m_size = other.m_size;
                }
            }

            /// @brief Defines the operator= for moving rvalue refrence
            /// @param other An rvalue refernce to a small vector
            /// @return A refrence to this small vector
            Small_vector& operator=(Small_vector&& other) noexcept {
                if(this != &other) {
                    if(on_heap()) {
                        delete[] m_data;
                        m_data = m_inline;
                        m_cap = N;
                    }

                    if(other.on_heap()) {
                        m_data = other.m_data;
                        m_cap = other.m_cap;
                        m_size = other.m_size;
                        other.m_data = other.m_inline;
                        other.m_cap = N;
                        other.m_size = 0;
                    }
                    else {
                        std::copy(other.m_data, other.m_data + other.m_size, m_inline);
                        m_size = other.m_size;
                    }
                }

                return *this;
            }

            /// @brief Destructor for the small vector
            ~Small_vector() {
                if(on_heap()) {
                    delete[] m_data;
                }
            }

            /// @brief Defines access operator[]
            /// @param i An std::size_t, index to access
            /// @return A reference to the element at the i:th position
            T& operator[](std::size_t i) {
                return m_data[i];
            }
            const T& operator[](std::size_t i) const {
                return m_data[i];
            }

            /// @brief Gets the element at the back of the small vector
            /// @return The last element
            T& back() {
                return m_data[m_size - 1];
            }
            const T& back() const {
                return m_data[m_size - 1];
            }

            /// @brief Calculate the size of the small vector
            /// @return The size
            std::size_t size() const {
                return m_size;
            }

            /// @brief Checks if small vector is empty
            /// @return True if empty, false otherwise
            bool empty() const {
                return m_size == 0;
            }

            /// @brief Gets the pointer to the beginning of the small vector
            /// @return The pointer
            T* begin() {
                return m_data;
            }
            const T* begin() const {
                return m_data;
            }

            /// @brief Gets the pointer to the end of the small vector (after
            /// last element)
            /// @return The pointer
            T* end() {
                return m_data + m_size;
            }
            const T* end() const {
                return m_data + m_size;
            }

            /// @brief Empties the small vector
            void clear() {
                m_size = 0;
            }

            /// @brief Adds element to the back of the small vector
            /// @param value The element to add
            void push_back(const T& value) {
                grow_to(m_size + 1);
                m_data[m_size++] = value;
            }

            /// @brief Allocates memeory for the small vector
            /// @param count The amount of elements to allocate for
            /// @param value The value to initialize the elements to
            void assign(std::size_t count, const T& value) {
                m_size = 0;
                grow_to(count);
                std::fill(m_data, m_data + count, value);
                m_size = count;
            }

            /// @brief Inserts elements into the small vector
            /// @param pos The pointer to the position of the small vector to start inserting
            /// @param count The number of elements to insert
            /// @param value The value of the elements to insert
            /// @return A pointer to the first element inserted
            T* insert(T* pos, std::size_t count, const T& value) {
                std::size_t index {static_cast<std::size_t>(pos - m_data)};
                grow_to(m_size + count);
                T* p {m_data + index};
                std::copy_backward(p, m_data + m_size, m_data + m_size + count);
                std::fill(p, p + count, value);
                m_size += count;
                return m_data + index;
            }

            T* insert(T* pos, const T& value) {
                return insert(pos, 1, value);
            }

            /// @brief Erases elements from the small vector
            /// @param first A pointer to the first element to erase
            /// @param last A pointer to the last element to erase (excluded)
            /// @return A pointer to the element previously at last
            T* erase(T* first, T* last) {
                std::size_t index {static_cast<std::size_t>(first - m_data)};
                std::size_t n {static_cast<std::size_t>(last - first)};
                std::copy(last, m_data + m_size, first);
                m_size -= n;
                return m_data + index;
            }

            T* erase(T* pos) {
                return erase(pos, pos + 1);
            }

            /// @brief Checks if two small vectors are identical
            /// @param a The first small vector
            /// @param b The second small vector
            /// @return True if identical, false otherwise
            friend bool operator==(const Small_vector& a, const Small_vector& b) {
                if(a.m_size != b.m_size) {
                    return false;
                }
                for(std::size_t i {}; i < a.m_size; ++i) {
                    if(a.m_data[i] != b.m_data[i]) {
                        return false;
                    }
                }
                return true;
            }
    };
}

#endif