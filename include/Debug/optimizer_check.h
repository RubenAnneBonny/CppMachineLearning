#ifndef OPTIMIZER_CHECK_H
#define OPTIMIZER_CHECK_H

#include <NN/optimizer.h>
#include <NN/parameter.h>
#include <LinAlg/tensor.h>
#include <Rand/random.h>

namespace Debug {
    struct Optimizer_result {
        bool passed = false;
        bool descent_check = false;
        bool parameter_coverage = false;
        bool zero_gradient = false;
    };

    template <NN::Optimizer<double> Opt>
    Optimizer_result optimizer_check(Opt optimizer) {
        Optimizer_result result;
        Rand::Random<double> random {};

        {
            Opt opt {optimizer};
            NN::Parameter<double> param {{3, 4}};
            param.normal(random, 0, 1);

            param.grad = 2 * param.value;

            LinAlg::Tensor<double> before = param.value.copy();
            std::vector<NN::Parameter<double>*> params {&param};
            opt.init(params);
            opt.step();

            LinAlg::Tensor<double> delta = param.value - before;
            double dot = (delta * param.grad).sum();
            result.descent_check = (dot < 0);
        }

        {
            Opt opt {optimizer};
            std::vector<NN::Parameter<double>> params {};

            for(int i {}; i < 5; ++i) {
                params.push_back(NN::Parameter<double>({3, 4}));
                params[i].normal(random, 0, 1);
                LinAlg::Tensor<double> A {{3, 4}};
                A.normal(random, 0, 1);
                params[i].grad = A;
            }

            std::vector<LinAlg::Tensor<double>> before {};

            for(int i {}; i < 5; ++i) {
                before.push_back(params[i].value.copy());
            }

            std::vector<NN::Parameter<double>*> pparams(5);
            for(int i {}; i < 5; ++i) {
                pparams[i] = &params[i];
            }

            opt.init(pparams);
            opt.step();

            result.parameter_coverage = true;

            for(int i {}; i < 5; ++i) {
                if(params[i].value == before[i]) {
                    result.parameter_coverage = false;
                }
            }
        }

        {
            Opt opt {optimizer};
            NN::Parameter<double> param {{3, 4}};
            param.normal(random, 0, 1);
            param.zero_grad();

            LinAlg::Tensor<double> before {param.value.copy()};
            std::vector<NN::Parameter<double>*> params {&param};
            opt.init(params);
            opt.step();

            result.zero_gradient = LinAlg::all_close<double>(param.value, before, 1e-7, 1e-7);
        }

        result.passed = result.zero_gradient && result.descent_check && result.parameter_coverage;

        return result;
    }   
}

#endif