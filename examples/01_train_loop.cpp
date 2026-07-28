#include <cml.h>
#include <iostream>
#include <iomanip>

int main() {
    // Minimal end-to-end. Initialize model -> train -> test

    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1000)};

    LinAlg::Tensor<double> train_X {data.inputs.slice(0, 800)};
    LinAlg::Tensor<double> train_Y {data.targets.slice(0, 800)};
    LinAlg::Tensor<double> test_X {data.inputs.slice(800, 1000)};
    LinAlg::Tensor<double> test_Y {data.targets.slice(800, 1000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model.init(random, data.inputs);

    std::vector<double> losses {model.train_loop(random, train_X, train_Y, 200, 32)};
    double test_loss {model.test_loop(test_X, test_Y)};
    
    std::cout << "Train loss: " << losses.back() << '\n';
    std::cout << "Test loss: " << test_loss << '\n';

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

    std::cout << std::fixed << std::setprecision(1);

    std::cout << "Train accuracy: " << accuracy(train_X, train_Y) << "%\n";
    std::cout << "Test accuracy: " << accuracy(test_X, test_Y) << "%\n";

    return 0;
}