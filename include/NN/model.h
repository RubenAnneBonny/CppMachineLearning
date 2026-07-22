#ifndef MODEL_H
#define MODEL_H

#include <LinAlg/tensor.h>
#include <NN/layer.h>
#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <Func/function.h>
#include <Data/dataloader.h>
#include <vector>
#include <NN/layer_base.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>

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
                if(target.get_extent(1) != m_layers.back()->get_nodes()) {
                    throw std::invalid_argument(
                        "Extent of second axis of target must match last layer shape " + 
                        std::to_string(m_layers.back()->get_nodes()) + 
                        " while target shape is " + 
                        static_cast<std::string>(target)
                    );
                }
            }

        public:
            /// @brief Constructor for the model
            /// @param loss_fn The loss function used to calculate the loss for the network
            /// @param optimizer The optimizer used to update weights based on their gradients
            Model(Loss loss_fn, Opt optimizer);

            /// @brief Collects all parameters of the model
            /// @return The parameters
            const std::vector<Parameter<T>*>& get_parameters() const {
                return m_parameters;
            }

            /// @brief Adds a layer to the neural network
            /// @tparam F The function type of the layer
            /// @tparam A The activation function type of the layer
            /// @param layer The layer to add
            /// @throws std::invalid_argument if network has already been initialized
            /// @throws std::invalid_argument if input size of layer dont match previous layer
            template <Func::Function<T> F,
                      Func::Activation_function<T> A>
            void add_layer(NN::Layer<T, F, A> layer);

            /// @brief Must run a init before training, initializes the network
            /// @throws std::invalid_argument if no layers were added before init
            void init();

            /// @brief Must run a init before training, initializes the network with random weights
            /// @param random The Random instance to initialize weights
            /// @param samples Samples to calculate optimal stddev for weights, not a must, but preferable use 5000 / (the least amount of nodes in layer)
            /// @param target_stddev The stddev we want for the output for each layer
            /// @param max_iters Maximum number of iterations to optimize for stddev
            /// @param tol The maximum tolerance for differnce between stddev from output from layer and target_stddev
            /// @throws std::invalid_argument if no layers were added before init
            /// @throws std::invalid_argument if the extent of the first axis of samples is 0
            void init(Rand::Random<T>& random, const LinAlg::Tensor<T>& samples, T target_stddev = 1, int max_iters = 5, T tol = 0.01, T damping = 0.9);

            /// @brief Does a forward pass through the network, saving necessary inputs
            /// @param X The input tensor to the network
            /// @return The networks raw prediction
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if the extent of the second axis of X don't match the first layer
            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            /// @brief Calculates all the intermediate outputs of each layer 
            /// @param input The tensor to pass through the network
            /// @return A vector of all the outputs of each layer in order
            /// @important Cannot be used as a normal forward pass
            std::vector<LinAlg::Tensor<T>> forward_capture(const LinAlg::Tensor<T>& input) const;
            
            /// @brief Calculates all the intermediate pre-activation outputs of each layer
            /// @param input The tensor to pass through the network
            /// @return A vector of all the outputs of each layer in order
            /// @important Cannot be used as a normal forward pass
            std::vector<LinAlg::Tensor<T>> pre_activation_capture(const LinAlg::Tensor<T>& input) const;

            /// @brief Calculates the loss
            /// @param target The target tensor
            /// @return The loss
            /// @throws std::invalid_argument if target rank isn't two
            /// @throws std::invalid_argument if target shape don't match output of network
            T calculate_loss(const LinAlg::Tensor<T>& target);

            /// @brief Resets the gradients for the optimizer
            void zero_grad();

            /// @brief Does a backpropagation through the network
            /// @param target The target tensor
            /// @throws std::invalid_argument if target rank isn't two
            /// @throws std::invalid_argument if target don't match output of network
            void backpropagation(const LinAlg::Tensor<T>& target);

            /// @brief The optimizer updates the weights
            void optimizer_step();

            /// @brief Saves the weights of the model
            /// @param path The file to write the weights to
            /// @throws std::invalid_argument if the file could not be opened
            void save_weights(const std::string& path) const;

            /// @brief Loads weights from a file to the model
            /// @param path The file to read the weights from
            /// @throws std::invalid_argument if file could not be opened
            /// @throws std::invalid_argument if the number of layers in model doesn't match the files
            /// @throws std::invalid_argument if a layer has a different number of nodes or weights than the save file
            /// @throws std::invalid_argument if the file was malformed or truncated
            void load_weights(const std::string& path);

            /// @brief A training loop for the neural network
            /// @param inputs The input tensor of shape (batch, input size)
            /// @param targets The target tensor of shape (batch, output size)
            /// @param epochs The number of epochs to train
            /// @return A vector of losses
            std::vector<T> train_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, Rand::Random<T>& random, int epochs, int batch_size);

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
                "At least one layer must be added to model before init"
            );
        } 

        m_initialized = true;

        m_optimizer.init(m_parameters);
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    void Model<T, Loss, Opt>::init(Rand::Random<T>& random, const LinAlg::Tensor<T>& samples, T target_stddev, int max_iters, T tol, T damping) {
        if(m_layers.empty()) {
            throw std::invalid_argument(
                "At least one layer must be added to model before init"
            );
        }

        int num_samples = samples.get_extent(0);

        if(num_samples == 0) {
            throw std::invalid_argument(
                "Must have at least one sample to calculate the stddev"
            );
        }

        for(auto* param : m_parameters) {
            param->normal(random, 0, 1);
        }

        for(int layer {}; layer < static_cast<int>(m_layers.size()); ++layer) {
            for(int iter {}; iter < max_iters; ++iter) {
                std::vector<T> values {};
                for(int sample {}; sample < num_samples; ++sample) {
                    LinAlg::Tensor<T> X {samples.row(sample).unsqueeze()};
                    for(int i {}; i <= layer; ++i) {
                        X = m_layers[i]->forward_pass(X);
                    }
                    X = m_layers[layer]->get_pre_activation();
                    for(int i {}; i < X.get_extent(1); ++i) {
                        values.push_back(X[{0, i}]);
                    }
                }
                int num_elements {static_cast<int>(values.size())};

                T sum {};
                for(T value : values) {
                    sum += value;
                }

                T mean {sum / num_elements};

                T variance {};
                for(T value : values) {
                    variance += (value - mean) * (value - mean);
                }
                variance /= num_elements;

                T stddev {std::sqrt(variance)};

                if(stddev < 1e-8) break;
                if(std::abs(stddev - target_stddev) < tol) break;
                m_parameters[layer]->value *= std::pow(target_stddev / stddev, damping);
            }
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

        if(out.get_extent(1) != m_layers[0]->get_input_nodes()) {
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
    std::vector<LinAlg::Tensor<T>> Model<T, Loss, Opt>::forward_capture(const LinAlg::Tensor<T>& input) const {
        std::vector<LinAlg::Tensor<T>> outputs;
        LinAlg::Tensor<T> X {input};
        for(auto& layer : m_layers) {
            X = layer->forward_pass(X);
            outputs.push_back(X);
        }
        return outputs;
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    std::vector<LinAlg::Tensor<T>> Model<T, Loss, Opt>::pre_activation_capture(const LinAlg::Tensor<T>& input) const {
        std::vector<LinAlg::Tensor<T>> outputs;
        LinAlg::Tensor<T> X {input};
        for(auto& layer : m_layers) {
            X = layer->forward_pass(X);
            outputs.push_back(layer->get_pre_activation());
        }
        return outputs;
    }    

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    T Model<T, Loss, Opt>::calculate_loss(const LinAlg::Tensor<T>& target) {
        if(target.get_rank() != 2) {
            throw std::invalid_argument(
                "Cannot perform loss calculations with target of shape " + 
                static_cast<std::string>(target) + 
                " since it isn't rank 2"
            );
        }

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
        if(target.get_rank() != 2) {
            throw std::invalid_argument(
                "Cannot perform backpropagation with target of shape " + 
                static_cast<std::string>(target) + 
                " since it isn't rank 2"
            );
        }
        
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
    void Model<T, Loss, Opt>::save_weights(const std::string& path) const {
        std::ofstream out(path);
        if(!out) {
            throw std::invalid_argument(
                "Save weights could not open " + 
                path
            );
        }

        out << std::setprecision(std::numeric_limits<T>::max_digits10);

        const std::vector<NN::Parameter<T>*>& params = get_parameters();
        out << params.size() << "\n";

        for(int param {}; param < static_cast<int>(params.size()); ++param) {
            const LinAlg::Tensor<T>& W {params[param]->value};
            int rows {W.get_extent(0)};
            int cols {W.get_extent(1)};

            out << param << " " << rows << " " << cols << "\n";

            for(int row {}; row < rows; ++row) {
                for(int col {}; col < cols; ++col) {
                    out << W[{row, col}] << " ";
                }
                out << "\n";
            }
        } 
    } 
    
    template <std::floating_point T,
            Func::Loss_function<T> Loss,
            NN::Optimizer<T> Opt>
    void Model<T, Loss, Opt>::load_weights(const std::string& path) {
        std::ifstream in(path);
        if(!in) {
            throw std::invalid_argument(
                "load weights could not open" + 
                path
            );
        }

        const std::vector<NN::Parameter<T>*>& params {get_parameters()};

        int num_parameters {};
        in >> num_parameters;
        if(num_parameters != static_cast<int>(params.size())) {
            throw std::invalid_argument(
                path + 
                " has weights for " + 
                std::to_string(num_parameters) + 
                " layers while the model has " + 
                std::to_string(static_cast<int>(params.size())) + 
                " layers"
            );
        }

        for(int param {}; param < num_parameters; ++param) {
            LinAlg::Tensor<T>& W {params[param]->value};

            int index {};
            int rows {};
            int cols {};
            in >> index >> rows >> cols;

            if(index != param) {
                throw std::invalid_argument(
                    "load weights expected layer " + 
                    std::to_string(param + 1) + 
                    " but next layer in load file is " + 
                    std::to_string(index + 1)
                );
            }

            if(rows != W.get_extent(0)) {
                throw std::invalid_argument(
                    "layer #" + 
                    std::to_string(index + 1) + 
                    " has " + 
                    std::to_string(W.get_extent(0)) + 
                    " nodes while the load file's layer has " + 
                    std::to_string(rows) + 
                    " nodes"
                );
            };

            if(cols != W.get_extent(1)) {
                throw std::invalid_argument(
                    "layer #" + 
                    std::to_string(index + 1) + 
                    " has " + 
                    std::to_string(W.get_extent(1)) + 
                    " weights pre node while the load file's layer has " + 
                    std::to_string(cols) + 
                    " weights per node"
                );
            }

            for(int row {}; row < rows; ++row) {
                for(int col {}; col < cols; ++col) {
                    in >> W[{row, col}];
                }
            }
        }

        if(in.fail()) {
            throw std::invalid_argument(
                "The path is malformed or truncated " + 
                path + 
                " couldnt load weights"
            );
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    std::vector<T> Model<T, Loss, Opt>::train_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, Rand::Random<T>& random, int epochs, int batch_size) {
        std::vector<T> losses {};
        losses.reserve(static_cast<std::size_t>(epochs));
        int num_inputs {inputs.get_extent(0)};

        Data::Dataloader<T> loader {random, inputs, targets, batch_size};

        for(int epoch {}; epoch < epochs; ++epoch) {
            T epoch_loss {};

            LinAlg::Tensor<T> input {{1}};
            LinAlg::Tensor<T> target {{1}};

            while(loader.next_batch(input, target)) {
                forward_pass(input);
                epoch_loss += calculate_loss(target);
                zero_grad();
                backpropagation(target);
                optimizer_step();
            }

            losses.push_back(epoch_loss / loader.get_num_batches());
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