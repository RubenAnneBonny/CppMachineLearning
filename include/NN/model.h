#ifndef CML_MODEL_H
#define CML_MODEL_H

#include <LinAlg/tensor.h>
#include <NN/layer.h>
#include <NN/parameter.h>
#include <NN/optimizer.h>
#include <Func/function.h>
#include <Data/data_loader.h>
#include <vector>
#include <NN/layer_base.h>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

namespace NN {
    /**
     * @brief A full neural network model
     *
     * @details Handles end-to-end neural network training and testing, either
     * with the the train_loop and test_loop or by using the exposed steps used
     * to train a network. 
     * 
     * @code 
     * NN::Layer<double, Func::Linear<double>, Func::ReLU<double>> layer {2, 1};
     * Func::MSE<double> loss_fn {};
     * NN::Adam<double> opt {};
     * NN::Model<double, Func::MSE<double>, NN::Adam<double>> model {loss_fn, opt};
     * model.add_layer(layer);
     * Rand::Random<double> random {};
     * LinAlg::Tensor<double> samples {{2000, 2}};
     * samples.normal(random, 0, 1);
     * model.init(random, samples);
     * LinAlg::Tensor<double> X {{5000, 2}};
     * LinAlg::Tensor<double> Y {{5000, 1}};
     * model.train(random, X, Y, 32);
     * @endcode
     */
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
            bool m_initialized;
            bool m_forward_passed;

            void check_target_shape(const LinAlg::Tensor<T>& target) const {
                if(target.get_rank() != 2) {
                    throw std::invalid_argument(
                        "Target must be of rank 2, currently target has shape " + 
                        static_cast<std::string>(target)
                    );
                }

                if(target.get_extent(1) != m_layers.back()->get_nodes()) {
                    throw std::invalid_argument(
                        "Extent of second axis of target must match last layer shape " + 
                        std::to_string(m_layers.back()->get_nodes()) + 
                        " while target shape is " + 
                        static_cast<std::string>(target)
                    );
                }

                if(target.get_extent(0) != m_store_prediction.get_extent(0)) {
                    throw std::invalid_argument(
                        "Batch size of target dont match the prediction of the target " + 
                        static_cast<std::string>(target)
                    );
                }
            }

            void check_input_shape(const LinAlg::Tensor<T>& input) const {
                if(input.get_rank() != 2) {
                    throw std::invalid_argument(
                        "Cannot use tensor of shape " + 
                        static_cast<std::string>(input) + 
                        " as input to network since its rank isn't 2"
                    );
                }

                if(input.get_extent(1) != m_layers[0]->get_input_nodes()) {
                    throw std::invalid_argument(
                        "Input tensor to network must be size (Batch, " + 
                        std::to_string(m_layers[0]->get_input_nodes()) + 
                        ") but input tensor was " + 
                        static_cast<std::string>(input)
                    );
                }
            }

            void require_initialized() const {
                if(!m_initialized) {
                    throw std::invalid_argument(
                        "Network must have been initialized with init() before use!"
                    );
                }
            }

            void require_forward_passed(std::string function) const {
                if(!m_forward_passed) {
                    throw std::invalid_argument(
                        "Forward pass was not perfomed prior to " +
                        function 
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
            /// @throws std::invalid_argument if model was already initialized
            /// @warning Only use this method if you manually set weights or use load_weights, otherwise model can't learn
            void init();

            /// @brief Must run a init before training, initializes the network with random weights
            /// @param random The Random instance to initialize weights
            /// @param samples Samples to calculate optimal stddev for weights, not a must, but preferable use 5000 / (the least amount of nodes in layer)
            /// @param target_stddev The stddev we want for the output for each layer
            /// @param max_iters Maximum number of iterations to optimize for stddev
            /// @param tol The maximum tolerance for differnce between stddev from output from layer and target_stddev
            /// @param damping Limits the step size towards correct stddev, should be in (0, 1]
            /// @throws std::invalid_argument if no layers were added before init
            /// @throws std::invalid_argument if model was already initialized
            void init(Rand::Random<T>& random, const LinAlg::Tensor<T>& samples, T target_stddev = T{1}, int max_iters = 5, T tol = T{0.01}, T damping = T{0.9});

            /// @brief Does a forward pass through the network, saving necessary inputs
            /// @param X The input tensor to the network
            /// @return The networks raw prediction
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if rank of X isn't two
            /// @throws std::invalid_argument if the extent of the second axis of X don't match the first layer
            LinAlg::Tensor<T> forward_pass(const LinAlg::Tensor<T>& X);

            /// @brief Does a forward pass through the network, without saving states
            /// @param X The input tensor to the network
            /// @return The networks raw prediction
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if rank of X isn't two
            /// @throws std::invalid_argument if the extent of the second axis of X don't match the first layer
            LinAlg::Tensor<T> forward_pass_stateless(const LinAlg::Tensor<T>& X) const;

            /// @brief Calculates all the intermediate outputs of each layer 
            /// @param input The tensor to pass through the network
            /// @return A vector of all the outputs of each layer in order
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @warning Cannot be used as a normal forward pass
            std::vector<LinAlg::Tensor<T>> forward_capture(const LinAlg::Tensor<T>& input) const;
            
            /// @brief Calculates all the intermediate pre-activation outputs of each layer
            /// @param input The tensor to pass through the network
            /// @return A vector of all the outputs of each layer in order
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @warning Cannot be used as a normal forward pass
            std::vector<LinAlg::Tensor<T>> pre_activation_capture(const LinAlg::Tensor<T>& input);

            /// @brief Calculates the loss
            /// @param target The target tensor
            /// @return The loss
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if target rank isn't two
            /// @throws std::invalid_argument if target shape don't match output of network
            T calculate_loss(const LinAlg::Tensor<T>& target);

            /// @brief Resets the gradients for the optimizer
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            void zero_grad();

            /// @brief Does a backpropagation through the network
            /// @param target The target tensor
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            /// @throws std::invalid_argument if target rank isn't two
            /// @throws std::invalid_argument if target don't match output of network
            void backpropagation(const LinAlg::Tensor<T>& target);

            /// @brief The optimizer updates the weights
            /// @throws std::invalid_argument if network hasn't been initialized with init()
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
            /// @param random A random instance used for random batching
            /// @param inputs The input tensor of shape (batch, input size)
            /// @param targets The target tensor of shape (batch, output size)
            /// @param epochs The number of epochs to train
            /// @param batch_size The batch size to use during training
            /// @return A vector of losses
            /// @throws std::invalid_argument if epochs is less than 1
            /// @throws std::invalid_argument if batch_size is less than 1
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            std::vector<T> train_loop(Rand::Random<T>& random, const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int epochs, int batch_size);

            /// @brief A testing loop for the neural network
            /// @param inputs The input tensor of shape (batch, input size)
            /// @param targets The target tensor of shape (batch, output size)
            /// @return The loss
            /// @throws std::invalid_argument if network hasn't been initialized with init()
            T test_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets) const;
    
            Model(const Model&) = delete;
            Model& operator=(const Model&) = delete;

            Model(Model&&) = default;
            Model& operator=(Model&&) = default;
    };

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    Model<T, Loss, Opt>::Model(Loss loss_fn, Opt optimizer)
        : m_loss_fn {loss_fn}
        , m_optimizer {optimizer}
        , m_store_prediction {{1, 1}}
        , m_initialized {false}
        , m_forward_passed {false}
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
        m_parameters.push_back(&holder->get_parameters());
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

        if(m_initialized) {
            throw std::invalid_argument(
                "Cant re-initialize a model"
            );
        }

        m_initialized = true;

        m_optimizer.init(m_parameters);
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    void Model<T, Loss, Opt>::init(Rand::Random<T>& random, const LinAlg::Tensor<T>& samples, T target_stddev, int max_iters, T tol, T damping) {
        int num_samples = samples.get_extent(0);
        
        init();

        for(auto* param : m_parameters) {
            param->normal(random, 0, 1);
        }

        LinAlg::Tensor<T> layer_input {samples.copy()};

        for(int layer {}; layer < static_cast<int>(m_layers.size()); ++layer) {
            for(int iter {}; iter < max_iters; ++iter) {
                m_layers[layer]->forward_pass(layer_input);
                LinAlg::Tensor<T> pre_activation {m_layers[layer]->get_pre_activation()};

                int batches {pre_activation.get_extent(0)};
                int nodes {pre_activation.get_extent(1)};
                int count {batches * nodes};

                T sum {};
                for(int b {}; b < batches; ++b) {
                    for(int i {}; i < nodes; ++i) {
                        sum += pre_activation[{b, i}];
                    }
                }

                T mean {sum / static_cast<T>(count)};

                T variance {};
                for(int b {}; b < batches; ++b) {
                    for(int i {}; i < nodes; ++i) {
                        T diff {pre_activation[{b, i}] - mean};
                        variance += diff * diff;
                    }
                }

                variance /= static_cast<T>(count);

                T stddev {std::sqrt(variance)};

                if(stddev < 1e-8) break;
                if(std::abs(stddev - target_stddev) < tol) break;
                m_parameters[layer]->value *= std::pow(target_stddev / stddev, damping);
            }

            layer_input = m_layers[layer]->forward_pass(layer_input);
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    LinAlg::Tensor<T> Model<T, Loss, Opt>::forward_pass(const LinAlg::Tensor<T>& X) {
        require_initialized();
        m_forward_passed = true;

        check_input_shape(X);

        LinAlg::Tensor<T> out {X};

        for(auto& layer : m_layers) {
            out = layer->forward_pass(out);
        }

        m_store_prediction = out.copy();

        return out;
    }

    template <std::floating_point T,
            Func::Loss_function<T> Loss,
            NN::Optimizer<T> Opt> 
    LinAlg::Tensor<T> Model<T, Loss, Opt>::forward_pass_stateless(const LinAlg::Tensor<T>& X) const {
        require_initialized();

        check_input_shape(X);

        LinAlg::Tensor<T> out {X};

        for(auto& layer : m_layers) {
            out = layer->forward_pass_stateless(out);
        }

        return out;
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    std::vector<LinAlg::Tensor<T>> Model<T, Loss, Opt>::forward_capture(const LinAlg::Tensor<T>& input) const {
        require_initialized();
       
        std::vector<LinAlg::Tensor<T>> outputs;
        LinAlg::Tensor<T> X {input};
        for(auto& layer : m_layers) {
            X = layer->forward_pass_stateless(X);
            outputs.push_back(X);
        }
        return outputs;
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    std::vector<LinAlg::Tensor<T>> Model<T, Loss, Opt>::pre_activation_capture(const LinAlg::Tensor<T>& input) {
        require_initialized();

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
        require_initialized();
        require_forward_passed("calculate_loss");

        check_target_shape(target);
        int batches {target.get_extent(0)};

        T loss {};

        for(int b {}; b < batches; ++b) {
            loss += m_loss_fn.loss(m_store_prediction.row(b).unsqueeze(), target.row(b).unsqueeze());
        }        

        loss /= static_cast<T>(batches);

        return loss;
    } 

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    void Model<T, Loss, Opt>::zero_grad() {
        require_initialized();

        for(auto& parameter : m_parameters) {
            parameter->zero_grad();
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt> 
    void Model<T, Loss, Opt>::backpropagation(const LinAlg::Tensor<T>& target) {
        require_initialized();
        require_forward_passed("backpropagation");
        
        if(target.get_rank() != 2) {
            throw std::invalid_argument(
                "Cannot perform backpropagation with target of shape " + 
                static_cast<std::string>(target) + 
                " since it isn't rank 2"
            );
        }
        
        check_target_shape(target);

        LinAlg::Tensor<T> dY {target.get_shape()};
        int batches {target.get_extent(0)};

        for(int b {}; b < batches; ++b) {
            LinAlg::Tensor<T> row_b {m_loss_fn.gradient(m_store_prediction.row(b).unsqueeze(), target.row(b).unsqueeze())};

            for(int i {}; i < row_b.get_extent(1); ++i) {
                dY[{b, i}] = row_b[{0, i}] / static_cast<T>(batches);
            }
        }

        for(auto it {m_layers.rbegin()}; it != m_layers.rend(); ++it) {
            dY = (*it)->backward_pass(dY);
        }
    }

    template <std::floating_point T,
              Func::Loss_function<T> Loss,
              NN::Optimizer<T> Opt>
    void Model<T, Loss, Opt>::optimizer_step() {
        require_initialized();

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
                "load weights could not open " + 
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
            }

            if(cols != W.get_extent(1)) {
                throw std::invalid_argument(
                    "layer #" + 
                    std::to_string(index + 1) + 
                    " has " + 
                    std::to_string(W.get_extent(1)) + 
                    " weights per node while the load file's layer has " + 
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
    std::vector<T> Model<T, Loss, Opt>::train_loop(Rand::Random<T>& random, const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets, int epochs, int batch_size) {
        if(epochs < 1) {
            throw std::invalid_argument(
                "Cannot perform a train loop with " + 
                std::to_string(epochs) + 
                " epochs since its less than 1"
            );
        }

        if(batch_size < 1) {
            throw std::invalid_argument(
                "Cannot perform a train loop with " + 
                std::to_string(batch_size) + 
                " batch_size since batch is less than 1"
            );
        }
        
        require_initialized();
        
        std::vector<T> losses {};
        losses.reserve(static_cast<std::size_t>(epochs));

        Data::Data_loader<T> loader {random, inputs, targets, batch_size};

        for(int epoch {}; epoch < epochs; ++epoch) {
            T epoch_loss {};

            LinAlg::Tensor<T> input {{1}};
            LinAlg::Tensor<T> target {{1}};

            while(loader.next_batch(random, input, target)) {
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
    T Model<T, Loss, Opt>::test_loop(const LinAlg::Tensor<T>& inputs, const LinAlg::Tensor<T>& targets) const {
        require_initialized();
        
        T loss {};
        int samples {inputs.get_extent(0)};

        for(int i {}; i < samples; ++i) {
            LinAlg::Tensor<T> prediction {forward_pass_stateless(inputs.row(i).unsqueeze())};
            loss += m_loss_fn.loss(prediction, targets.row(i).unsqueeze());
        }

        loss /= static_cast<T>(samples);

        return loss;
    }
} 

#endif