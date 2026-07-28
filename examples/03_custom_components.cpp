#include <cml.h>
#include <iostream>

// Custom activation function following the rules set by the activation
// function concept
template <std::floating_point T>
class Leaky_relu {
    public:
        static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
            LinAlg::Tensor<T> out {X.copy()};
            auto leaky{
                [](T a)
                {
                    return a > T{0} ? a : T{0.01} * a;
                }
            };
            out.elementwise(leaky);
            return out;
        }

        static LinAlg::Tensor<T> derivative(const LinAlg::Tensor<T>& X) {
            LinAlg::Tensor<T> out {X.copy()};
            auto leaky_derivative{
                [](T a)
                {
                    return a > T{0} ? T{1} : T{0.01};
                }
            };
            out.elementwise(leaky_derivative);
            return out;
        }
};

// Custom optimizer following the rules set by the optimizer concept
template <std::floating_point T>
class Momentum {
    private:
        T m_lr;
        T m_mu;
        std::vector<NN::Parameter<T>*> m_params;
        std::vector<LinAlg::Tensor<T>> m_velocity;
    
    public:
        explicit Momentum(T lr, T mu = T{0.9})
            : m_lr {lr}
            , m_mu {mu}
        {}

        void init(std::vector<NN::Parameter<T>*>& params) {
            m_params = params;
            m_velocity.clear();
            for(auto* p : m_params) {
                m_velocity.push_back(LinAlg::Tensor<T>{p->value.get_shape()});
            }
        }

        void step() {
            for(std::size_t i {}; i < m_params.size(); ++i) {
                m_velocity[i] = m_velocity[i] * m_mu - m_params[i]->grad * m_lr;
                m_params[i]->value += m_velocity[i];
            }
        }
};

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_moons<double>(random, 1000)};

    LinAlg::Tensor<double> train_X {data.inputs.slice(0, 800)};
    LinAlg::Tensor<double> train_Y {data.targets.slice(0, 800)};
    LinAlg::Tensor<double> test_X {data.inputs.slice(800, 1000)};
    LinAlg::Tensor<double> test_Y {data.targets.slice(800, 1000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    Momentum<double> opt {0.01};

    // When creating custom functions and optimizer, check them with the
    // functions from the Debug library
    Debug::Optimizer_result opt_result {Debug::optimizer_check(opt)};
    Debug::GradCheck::Activation_result act_result {Debug::GradCheck::activation<Leaky_relu<double>>(4)};

    if(opt_result.passed) {
        std::cout << "Momentum optimizer passed common edge cases" << std::endl;
    }
    else {
        std::cout << "Momentum optimizer did not pass common edge cases" << std::endl;
    }

    if(act_result.passed) {
        std::cout << "The derivative of the leaky relu activation function matches its numerical value" << std::endl;
    }
    else {
        std::cout << "The derivative of the leaky relu fails numerical tests" << std::endl;
    }

    NN::Model<double, Func::Softmax_cross_entropy<double>, Momentum<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Leaky_relu<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Leaky_relu<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model.init(random, data.inputs);

    std::vector<double> losses {model.train_loop(random, train_X, train_Y, 200, 32)};
    double test_loss {model.test_loop(test_X, test_Y)};

    // Accuracy helper
    auto accuracy {
        [&](const LinAlg::Tensor<double>& X, const LinAlg::Tensor<double>& Y) {
            LinAlg::Tensor<double> logits {model.forward_pass_stateless(X)};
            int correct {};
            for(int i {}; i < X.get_extent(0); ++i) {
                if(logits.row(i).argmax()[0] == Y.row(i).argmax()[0]) {
                    ++correct;
                }
            }
            return 100.0 * correct / X.get_extent(0);
        }
    };

    double train_acc {accuracy(train_X, train_Y)};
    double test_acc {accuracy(test_X, test_Y)};

    std::cout << "Train loss: " << losses.back() << "\n";
    std::cout << "Train acc: " << train_acc << "%\n";
    std::cout << "Test loss: " << test_loss << "\n";
    std::cout << "Test acc: " << test_acc << "%\n";

    return 0;
}