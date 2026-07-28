#include <cml.h>
#include <iostream>

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1000)};

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
    model_save.train_loop(random, data.inputs.slice(0, 800), data.targets.slice(0, 800), 200, 32);
    model_save.save_weights(path);

    // Loading part:
    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    // Same layers as in model_save
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model.init();
    model.load_weights(path);
    double test_loss {model.test_loop(data.inputs.slice(800, 1000), data.targets.slice(800, 1000))};

    std::cout << "Test loss: " << test_loss << '\n';

    return 0;
}