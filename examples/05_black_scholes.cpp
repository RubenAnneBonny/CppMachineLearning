#include <cml.h>
#include <iostream>

int main() {
    // Black-Scholes, learn call price from [S/K, T, sigma, r]

    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_black_scholes<double>(random, 1000)};

    Func::MSE<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::MSE<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{4, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 1});
    model.init(random, data.inputs);

    std::vector<double> losses {model.train_loop(random, data.inputs.slice(0, 800), data.targets.slice(0, 800), 200, 32)};
    double test_loss {model.test_loop(data.inputs.slice(800, 1000), data.targets.slice(800, 1000))};

    std::cout << "Final train loss: " << losses.back() << std::endl;
    std::cout << "Test loss " << test_loss;

    return 0;
}