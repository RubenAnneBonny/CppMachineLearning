#include <cml.h>
#include <iostream>

int main() {
    // Minimal end-to-end. Initialize model -> train -> test

    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    // Output layer uses no_activation since Softmax_cross_entropy takes raw logits
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model.init(random, data.inputs);

    std::vector<double> losses {model.train_loop(random, data.inputs.slice(0, 800), data.targets.slice(0, 800), 200, 32)};
    double test_loss {model.test_loop(data.inputs.slice(800, 1000), data.targets.slice(800, 1000))};

    std::cout << "Final train loss: " << losses.back() << std::endl;
    std::cout << "Test loss " << test_loss;

    return 0;
}