#include <cml.h>
#include <iostream>

int main() {
    // Black-Scholes, learn call price from [S/K, T, sigma, r]

    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_black_scholes<double>(random, 1000)};

    LinAlg::Tensor<double> train_X {data.inputs.slice(0, 800)};
    LinAlg::Tensor<double> train_Y {data.targets.slice(0, 800)};
    LinAlg::Tensor<double> test_X {data.inputs.slice(800, 1000)};
    LinAlg::Tensor<double> test_Y {data.targets.slice(800, 1000)};

    Func::MSE<double> loss_fn {};
    NN::Adam<double> opt {};
    NN::Model<double, Func::MSE<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{4, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 1});
    model.init(random, data.inputs);

    std::vector<double> losses {model.train_loop(random, train_X, train_Y, 200, 32)};
    double test_loss {model.test_loop(test_X, test_Y)};

    std::cout << "Train: {MSE: " << losses.back() << ", RMSE: " << std::sqrt(losses.back()) << "}\n";
    std::cout << "Test: {MSE: " << test_loss << ", RMSE: " << std::sqrt(test_loss) << "}\n";

    std::cout << "\nA few concreate examples: \n";
    for(int i {}; i < 5; ++i) {
        LinAlg::Tensor<double> pred {model.forward_pass_stateless(test_X.row(i).unsqueeze())};
        std::cout << "\nTarget call price: " << test_Y[{i, 0}] << "\n";
        std::cout << "Predicted call prices: " << pred[{0, 0}] << "\n";
    }

    return 0;
}