#include <cml.h>
#include <iostream>

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1000)};

    LinAlg::Tensor<double> train_X {data.inputs.slice(0, 800)};
    LinAlg::Tensor<double> train_Y {data.targets.slice(0, 800)};
    LinAlg::Tensor<double> test_X {data.inputs.slice(800, 1000)};
    LinAlg::Tensor<double> test_Y {data.targets.slice(800, 1000)};

    // Saving part:
    std::string path {"model_04.weights"};
    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model_save {loss_fn, opt};
    model_save.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model_save.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model_save.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model_save.init(random, data.inputs);
    model_save.train_loop(random, train_X, train_Y, 200, 32);

    // Accuracy helper
    auto accuracy {
        [&](const LinAlg::Tensor<double>& X, const LinAlg::Tensor<double>& Y, auto& model) {
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

    double train_loss_before {model_save.test_loop(train_X, train_Y)};
    double test_loss_before {model_save.test_loop(test_X, test_Y)};
    double train_acc_before {accuracy(train_X, train_Y, model_save)};
    double test_acc_before {accuracy(test_X, test_Y, model_save)};

    model_save.save_weights(path);

    // Loading part:
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model_load {loss_fn, opt};
    // Same layers as in model_save
    model_load.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model_load.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    model_load.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model_load.init();
    model_load.load_weights(path);

    double train_loss_after {model_load.test_loop(train_X, train_Y)};
    double test_loss_after {model_load.test_loop(test_X, test_Y)};
    double train_acc_after {accuracy(train_X, train_Y, model_load)};
    double test_acc_after {accuracy(test_X, test_Y, model_load)};

    std::cout << "Before loading:\n";
    std::cout << "Train loss: " << train_loss_before << "\n";
    std::cout << "Train acc: " << train_acc_before << "%\n";
    std::cout << "Test loss: " << test_loss_before << "\n";
    std::cout << "Test acc: " << test_acc_before << "%\n";

    std::cout << "\nAfter loading:\n";
    std::cout << "Train loss: " << train_loss_after << "\n";
    std::cout << "Train acc: " << train_acc_after << "%\n";
    std::cout << "Test loss: " << test_loss_after << "\n";
    std::cout << "Test acc: " << test_acc_after << "%\n";

    return 0;
}