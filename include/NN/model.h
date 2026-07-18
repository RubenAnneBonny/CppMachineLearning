#ifndef MODEL_H
#define MODEL_H

#include <LinAlg/tensor.h>
#include <NN/layer.h>
#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <Func/function.h>
#include <vector>
#include <NN/layer_base.h>
#include <string>
#include <stdexcept>

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
            bool m_initialized {false};

            void check_target_shape(const LinAlg::Tensor<T>& target) const {
                if(target.get_extent(0) != 1 || target.get_extent(1) != m_layers.back()->get_nodes()) {
                    throw std::invalid_argument(
                        "Target must match last layer shape (1, " + 
                        std::to_string(m_layers.back()->get_nodes()) + 
                        ") while target shape is " + 
                        static_cast<std::string>(target)
                    );
                }
            }

        public:
            /// @brief Constructor for the model
            /// @param loss_fn The loss function used to calculate the loss for the network
            /// @param optimizer The optimizer used to update weights based on their gradients
            Model(Loss loss_fn, Opt optimizer);

            /// @brief Adds a layer to the neural network
            /// @tparam F The function type of the layer
            /// @tparam A The activation function type of the layer
            /// @param layer The layer to add
            /// @throws std::invalid_argument if network has already been initialized
            /// @throws std::invalid_argument if input size of layer dont match previous layer
            template <Func::Function<T> F,
                      Func::Activation_function<T> A>
            void add_layer(NN::Layer<T, F, A> layer);

            /// @brief Must run before training, initializes the network
            void init();

            /// @brief Does a forward pass through the network, saving necessary inputs
            /// @param X The input tensor to the network
            /// @return The networks raw prediction
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if the shape of X don't match the first layer
            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            /// @brief Calculates the loss
            /// @param target The target tensor
            /// @return The loss
            /// @throws std::invalid_argument if target shape don't match output of network
            T calculate_loss(const LinAlg::Tensor<T>& target);

            /// @brief Resets the gradients for the optimizer
            void zero_grad();

            /// @brief Does a backpropagation through the network
            /// @param target The target tensor
            /// @throws std::invalid_argument if target don't match output of network
            void backpropagation(const LinAlg::Tensor<T>& target);

            /// @brief The optimizer updates the weights
            void optimizer_step();

            /// @brief A training loop for the neural network
            /// @param inputs The input tensor of shape (batch, input size)
            /// @param targets The target tensor of shape (batch, output size)
            /// @param epochs The number of epochs to train
            /// @return A vector of losses
            std::vector<T> train_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int epochs);

            /// @brief A testing loop for the neural network
            /// @param inputs The input tensor of shape (batch, input size)
            /// @param targets The target tensor of shape (batch, output size)
            /// @return The loss
            T test_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets);
    };

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    Model<T, Loss, Opt>::Model(Loss loss_fn, Opt optimizer)
        : m_loss_fn {loss_fn}
        , m_optimizer {optimizer}
        , m_store_prediction {{1, 1}}
    {}

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    template <Func::Function<T> F,
              Func::Activation_function<T> A>
    void Model<T, Loss, Opt>::add_layer(NN::Layer<T, F, A> layer) {
        if(m_initialized) {
            throw std::invalid_argument(
                "Cannot add layers after the network have been initialized"
            );
        }

        if(!m_layers.empty()) {
            int prev_layer_size {m_layers.back()->get_nodes()};
            int layer_input_size {layer.get_input_nodes()};

            if(prev_layer_size != layer_input_size) {
                throw std::invalid_argument(
                    "Input sizes for layer dont match previous layer. Previous layer has " + 
                    std::to_string(prev_layer_size) + 
                    " nodes, while you're trying to add layer with input size of " + 
                    std::to_string(layer_input_size) + 
                    "."
                );
            }
        }

        auto holder {std::make_unique<NN::Layer_holder<T, NN::Layer<T, F, A>>>(std::move(layer))};
        m_parameters.push_back(&holder->parameters());
        m_layers.push_back(std::move(holder));
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    void Model<T, Loss, Opt>::init() {
        if(m_layers.empty()) {
            throw std::invalid_argument(
                "Layers must be added to model before init()"
            );
        } 

        m_initialized = true;

        m_optimizer.init(m_parameters);
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    LinAlg::Tensor<T> Model<T, Loss, Opt>::forward_pass(const LinAlg::Tensor<T>& X) {
        if(!m_initialized) {
            throw std::invalid_argument(
                "Network must have been initialized with init() before use!"
            );
        }

        LinAlg::Tensor<T> out {X};

        if(out.get_rank() != 2) {
            throw std::invalid_argument(
                "Cannot use tensor of shape " + 
                static_cast<std::string>(out) + 
                " since its rank isn't 2"
            );
        }

        if(out.get_extent(0) != 1 || out.get_extent(1) != m_layers[0]->get_input_nodes()) {
            throw std::invalid_argument(
                "Input tensor to forward pass must be size (1, " + 
                std::to_string(m_layers[0]->get_input_nodes()) + 
                ") but input tensor of forward pass was " + 
                static_cast<std::string>(out)
            );
        }

        for(auto& layer : m_layers) {
            out = layer->forward_pass(out);
        }

        m_store_prediction = out;

        return out;
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    T Model<T, Loss, Opt>::calculate_loss(const LinAlg::Tensor<T>& target) {
        check_target_shape(target);
        
        m_store_loss = m_loss_fn.loss(m_store_prediction, target);

        return m_store_loss;
    } 

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    void Model<T, Loss, Opt>::zero_grad() {
        for(auto& parameter : m_parameters) {
            parameter->zero_grad();
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    void Model<T, Loss, Opt>::backpropagation(const LinAlg::Tensor<T>& target) {
        check_target_shape(target);

        LinAlg::Tensor<T> dY {m_loss_fn.gradient(m_store_prediction, target)};

        for(auto it {m_layers.rbegin()}; it != m_layers.rend(); ++it) {
            dY = (*it)->backward_pass(dY);
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    void Model<T, Loss, Opt>::optimizer_step() {
        m_optimizer.step();
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    std::vector<T> Model<T, Loss, Opt>::train_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int epochs) {
        std::vector<T> losses {};
        losses.reserve(static_cast<std::size_t>(epochs));
        int num_inputs {inputs.get_extent(0)};

        for(int epoch {}; epoch < epochs; ++epoch) {
            T epoch_loss {};

            for(int i {}; i < num_inputs; ++i) {
                LinAlg::Tensor<T> input {inputs.row(i).unsqueeze()};
                LinAlg::Tensor<T> target {targets.row(i).unsqueeze()};

                forward_pass(input);
                epoch_loss += calculate_loss(target);
                zero_grad();
                backpropagation(target);
                optimizer_step();
            }

            losses.push_back(epoch_loss / static_cast<T>(num_inputs));
        }

        return losses;
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    T Model<T, Loss, Opt>::test_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets) {
        T loss {};
        int num_inputs {inputs.get_extent(0)};

        for(int i {}; i < num_inputs; ++i) {
            LinAlg::Tensor<T> input {inputs.row(i).unsqueeze()};
            LinAlg::Tensor<T> target {targets.row(i).unsqueeze()};

            forward_pass(input);
            loss += calculate_loss(target);
        }

        loss /= static_cast<T>(num_inputs);

        return loss;
    }
} 

#endif