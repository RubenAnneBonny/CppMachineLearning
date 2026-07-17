#ifndef MODEL_H
#define MODEL_H

#include <LinAlg/tensor.h>
#include <NN/layer.h>
#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <Func/function.h>
#include <vector>
#include <NN/layer_base.h>

namespace NN {
    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    class Model {
        private:
            Loss m_loss_fn;
            Opt m_optimizer;
            std::vector<std::unique_ptr<Layer_base<T>>> m_layers;
            std::vector<NN::Parameter<T>*> m_parameters;
            LinAlg::Tensor<T> m_store_prediction;
            T m_store_loss;

        public:
            Model(Loss loss_fn, Opt optimizer)
                : m_loss_fn {loss_fn}
                , m_optimizer {optimizer}
                , m_store_prediction {{1, 1}}
            {}

            template <Func::Function<T> F,
                      Func::Activation_function<T> A>
            void add_layer(NN::Layer<T, F, A> layer) {
                auto holder {std::make_unique<NN::Layer_holder<T, NN::Layer<T, F, A>>>(std::move(layer))};
                m_parameters.push_back(&holder->parameters());
                m_layers.push_back(std::move(holder));
            }

            void init() {
                m_optimizer.init(m_parameters);
            }

            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X) {
                LinAlg::Tensor<T> out {X};

                for(auto& layer : m_layers) {
                    out = layer->forward_pass(out);
                }

                m_store_prediction = out;

                return out;
            }

            T calculate_loss(const LinAlg::Tensor<T>& target) {
                m_store_loss = m_loss_fn.loss(m_store_prediction, target);

                return m_store_loss;
            }

            void zero_grad() {
                for(auto& parameter : m_parameters) {
                    parameter->zero_grad();
                }
            }

            void backpropagation(const LinAlg::Tensor<T>& target) {
                LinAlg::Tensor<T> dY {m_loss_fn.gradient(m_store_prediction, target)};

                for(auto it {m_layers.rbegin()}; it != m_layers.rend(); ++it) {
                    dY = (*it)->backward_pass(dY);
                }
            }

            void step() {
                m_optimizer.step();
            }
    };
} 

#endif