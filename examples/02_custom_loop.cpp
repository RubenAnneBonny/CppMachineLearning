#include <cml.h>
#include <iostream>

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 1000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 3});
    model.init(random, data.inputs);

    Data::Data_loader<double> loader {random, data.inputs.slice(0, 2400), data.targets.slice(0, 2400), 32};

    // X and Y are placeholders, will be replaced by loader
    LinAlg::Tensor<double> X {{1}};
    LinAlg::Tensor<double> Y {{1}};
    for(int epoch {}; epoch < 200; ++epoch) {
        // Custom loop. forward pass -> zero_grad -> backpropagation -> optimizer_step
        while(loader.next_batch(random, X, Y)) {
            model.forward_pass(X);
            model.zero_grad();
            model.backpropagation(Y);
            model.optimizer_step();
        }

        if(epoch % 20 == 0) {
            std::cout << "Test loss at epoch " << epoch << " = " << model.test_loop(data.inputs.slice(2400, 3000), data.targets.slice(2400, 3000)) << std::endl;
        }
    }

    return 0;
}