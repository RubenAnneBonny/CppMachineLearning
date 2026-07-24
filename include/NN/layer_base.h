#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include <memory>
#include <concepts>
#include <LinAlg/tensor.h>
#include <NN/parameter.h>

namespace NN {
    /// @brief These classes purpose is to allow vector of layers with different functions in model.h 

    template <std::floating_point T>
    class Layer_base {
        public:
            virtual ~Layer_base() = default;

            virtual LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X) = 0;
            virtual LinAlg::Tensor<T> forward_pass_stateless(const LinAlg::Tensor<T>& X) const = 0;
            virtual LinAlg::Tensor<T> backward_pass(const LinAlg::Tensor<T>& dY) = 0;
            virtual Parameter<T>& parameters() = 0;
            virtual int get_nodes() const = 0;
            virtual int get_input_nodes() const = 0;
            virtual LinAlg::Tensor<T> get_pre_activation() const = 0;
    };

    template <std::floating_point T, 
              typename L>
    class Layer_holder final : public Layer_base<T> {
        private:
            L m_layer;
        
        public:
            explicit Layer_holder(L layer) 
                : m_layer {std::move(layer)}
            {}

            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X) override {
                return m_layer.forward_pass(X);
            }

            LinAlg::Tensor<T> forward_pass_stateless(const LinAlg::Tensor<T>& X) const override {
                return m_layer.forward_pass_stateless(X);
            }

            LinAlg::Tensor<T> backward_pass(const LinAlg::Tensor<T>& dY) override {
                return m_layer.backward_pass(dY);
            }

            Parameter<T>& parameters() override {
                return m_layer.weights;
            }

            int get_nodes() const override {
                return m_layer.get_nodes();
            }

            int get_input_nodes() const override {
                return m_layer.get_input_nodes();
            }

            LinAlg::Tensor<T> get_pre_activation() const override {
                return m_layer.get_pre_activation();
            }
    };
}

#endif