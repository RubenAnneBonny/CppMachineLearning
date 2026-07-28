#include <cml.h>
#include <iostream>

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 1000)};

    LinAlg::Tensor<double> train_X {data.inputs.slice(0, 2400)};
    LinAlg::Tensor<double> train_Y {data.targets.slice(0, 2400)};
    LinAlg::Tensor<double> test_X {data.inputs.slice(2400, 3000)};
    LinAlg::Tensor<double> test_Y {data.targets.slice(2400, 3000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 3});
    model.init(random, data.inputs);

    Data::Data_loader<double> loader {random, train_X, train_Y, 32};

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

    // X and Y are placeholders, will be replaced by loader
    LinAlg::Tensor<double> X {{1}};
    LinAlg::Tensor<double> Y {{1}};
    for(int epoch {}; epoch < 201; ++epoch) {
        // Custom loop. forward pass -> zero_grad -> backpropagation -> optimizer_step
        while(loader.next_batch(random, X, Y)) {
            model.forward_pass(X);
            model.zero_grad();
            model.backpropagation(Y);
            model.optimizer_step();
        }

        if(epoch % 40 == 0) {
            double train_loss {model.test_loop(train_X, train_Y)};
            double test_loss {model.test_loop(test_X, test_Y)};
            double train_acc {accuracy(train_X, train_Y)};
            double test_acc {accuracy(test_X, test_Y)};

            std::cout << "Epoch " << epoch << "\n";
            std::cout << "Train: {Loss: " << train_loss << ", Acc: " << train_acc << "%}\n";
            std::cout << "Test: {Loss: " << test_loss << ", Acc: " << test_acc << "%}\n";
            std::cout << "\n";
        }
    }

    return 0;
}