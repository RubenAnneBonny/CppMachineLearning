# CppMachineLearning

**A header-only neural network library written from scratch in C++20. It has view-based tensors, layer components you plug in through C++20 concepts, and a public numerical gradient checker for verifying your own components.**

[![CI](https://github.com/RubenAnneBonny/CppMachineLearning/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenAnneBonny/CppMachineLearning/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Header-only](https://img.shields.io/badge/header--only-yes-success.svg)
![Tests](https://img.shields.io/badge/tests-199%20passing-brightgreen.svg)

There are no external dependencies and nothing to build for the library itself. You include the headers and use it. The tests pull in GoogleTest, but only when you build with them.

## What's in it

- **View-based tensors.** `LinAlg::Tensor<T>` is a shared pointer to a flat buffer plus a shape, strides and an offset. Copying a tensor copies that handle, not the data underneath. Slicing, transposing and broadcasting return views without copying anything, and `.copy()` is there for when you actually want new memory.
- **Broadcasting** on element-wise operations, following the same rules as NumPy.
- **Concepts for the pluggable parts.** Node functions, activations, losses and optimizers are each defined by a concept. Any type that meets the contract works inside `Layer` and `Model` directly, so there is no inheritance and no virtual calls in the math.
- **Functions don't own their weights.** The layer owns the parameters and the model owns the layers. Keeping the weights out of the function is what lets the concepts stay small.
- **Numerical checks for your own code.** If you write a component you can check its analytical derivative against a finite-difference gradient before you rely on it. There are four checkers, one per kind of component (see below).
- **Included components.** `Linear` layers, `ReLU` / `Sigmoid` / `No_activation`, `MSE` and a fused `Softmax_cross_entropy`, `Gradient_descent` and `Adam`, LSUV-style weight initialisation, weight save/load, a shuffling data loader, and generators for circles, moons, spirals and Black-Scholes data.

## Quick start

A model with two hidden layers, trained and evaluated on a synthetic `make_circles` dataset:

```cpp
#include <cml.h>
#include <iostream>
#include <vector>

int main() {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1000)};

    Func::Softmax_cross_entropy<double> loss_fn {};
    NN::Adam<double> opt {};

    NN::Model<double, Func::Softmax_cross_entropy<double>, NN::Adam<double>> model {loss_fn, opt};
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{2, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::ReLU<double>>{16, 16});
    model.add_layer(NN::Layer<double, Func::Linear<double>, Func::No_activation<double>>{16, 2});
    model.init(random, data.inputs);

    std::vector<double> losses {
        model.train_loop(random, data.inputs.slice(0, 800), data.targets.slice(0, 800), 200, 32)
    };
    double test_loss {
        model.test_loop(data.inputs.slice(800, 1000), data.targets.slice(800, 1000))
    };

    std::cout << "Final train loss: " << losses.back() << '\n';
    std::cout << "Test loss: " << test_loss << '\n';
}
```

## Tests

The library is checked by 199 tests that run on Ubuntu and Windows through GitHub Actions. Every `throw` in the library has a test that triggers it, and the numerical routines (optimizer steps, loss gradients, forward passes) are checked against values worked out by hand rather than just checked for not crashing.

The whole thing compiles clean under `-Wall -Wextra -Wconversion -Wsign-conversion -Werror` on GCC and Clang, and `/W4 /WX` on MSVC. CI treats warnings as errors, so it stays that way. 

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Writing your own components

Every part of a model is a type that satisfies a concept, so adding your own means writing a small class. Once you have, you can check it numerically:

- node/weight function: `Debug::GradCheck::function<F>(input_size)`
- activation: `Debug::GradCheck::activation<A>(input_size)`
- loss: `Debug::GradCheck::loss<L>(sample_target)`
- optimizer: `Debug::optimizer_check<Opt>(optimizer)`

Each returns a small struct with a `passed` flag. Here is a Leaky ReLU checked against its numerical gradient:

```cpp
#include <cml.h>
#include <iostream>

template <std::floating_point T>
class Leaky_relu {
    public:
        static LinAlg::Tensor<T> activate(const LinAlg::Tensor<T>& X) {
            LinAlg::Tensor<T> out {X.copy()};
            out.elementwise([](T a) { return a > T{0} ? a : T{0.01} * a; });
            return out;
        }

        static LinAlg::Tensor<T> derivative(const LinAlg::Tensor<T>& X) {
            LinAlg::Tensor<T> out {X.copy()};
            out.elementwise([](T a) { return a > T{0} ? T{1} : T{0.01}; });
            return out;   
        }
};

int main() {
    Debug::GradCheck::Activation_result result {
        Debug::GradCheck::activation<Leaky_relu<double>>(4)
    };

    std::cout << (result.passed
        ? "Leaky ReLU derivative matches its numerical gradient.\n"
        : "Leaky ReLU derivative is wrong.\n");
}
```

## Examples 

Five programs in `examples/`, each covering something different:

- **01** is the shortest full run: build a model, `train_loop`, evaluate. Trained on `make_circles`.
- **02** writes the training loop by hand using the forward pass, backpropagation and optimizer step calls instead of `train_loop`. Trained on `make_spirals`.
- **03** defines a custom `Leaky_relu` activation and a `Momentum` optimizer, runs both through the `Debug` checkers, then trains on `make_moons`.
- **04** saves a trained model's weights, loads them into a new model, and shows the predictions match.
- **05** is a Black-Scholes option pricer. The training data comes from the exact formula, and the network learns to price across moneyness, maturity, volatility and rate.


## Building and using it

You need a C++20 compiler (GCC 10+, Clang 12+ or MSVC 2019+) and CMake 3.14 or newer.

Build and run the tests:

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests and examples are on by default. To turn them off:

```bash
cmake -B build -DCML_BUILD_TESTS=OFF -DCML_BUILD_EXAMPLES=OFF
```

To use the library in your own CMake project, link against the `CML::CML` target and include `<cml.h>`. The target carries the include path and the C++20 requirement for you.

Vendored as a subdirectory of your repository: 

```cmake
add_subdirectory(external/CppMachineLearning)

target_link_libraries(your_target PRIVATE CML::CML)
```

Or fetched at configure time:

```
include(FetchContent)
FetchContent_Declare(
    cml
    GIT_REPOSITORY https://github.com/RubenAnneBonny/CppMachineLearning.git
    GIT_TAG main
)
FetchContent_MakeAvailable(cml)

target_link_libraries(your_target PRIVATE CML::CML)
```

The library's tests and examples only build when CppMachineLearning is the top-level project, so pulling it into you build won't drag them along.

## A few design decisions

- Tensors are handles over shared storage, so copies are cheap and views are $\mathcal O(1)$. Use `.copy()` when you want the real new memory.
- Functions, activations and losses work on one sample at a time. The model handles the batch loop and averaging. This keeps each component's concept small.
- `Softmax_cross_entropy` takes raw logits and does the softmax itself, which is why the output layer before it uses `No_activation`. It is steadier numerically than a separate softmax step.
- `init(random, samples, ...)` scales each layer's weights to hit a target pre-activation spread on real input data instead of using a fixed rule.
- The architecture lives in the C++ types, so it is not serialised. You rebuild the same model in code and `load_weights` checks the shapes line up before loading.