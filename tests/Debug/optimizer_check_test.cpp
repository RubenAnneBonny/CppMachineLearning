#include <Debug/optimizer_check.h>
#include <NN/parameter.h>
#include <gtest/gtest.h>

namespace {
    template <std::floating_point T>
    class Obvious_optimizer {
        private:
            std::vector<NN::Parameter<T>*> m_params;

        public: 
            Obvious_optimizer() 
            {}

            void init(std::vector<NN::Parameter<T>*>& params) {
                m_params = params;
            }

            void step() {
                for(auto* param : m_params) {
                    param->value -= param->grad;
                }
            }
    };

    template <std::floating_point T>
    class Wrong_direction {
        private:
            std::vector<NN::Parameter<T>*> m_params;

        public: 
            Wrong_direction() 
            {}

            void init(std::vector<NN::Parameter<T>*>& params) {
                m_params = params;
            }

            void step() {
                for(auto* param : m_params) {
                    param->value += param->grad;
                }
            }
    };

    template <std::floating_point T>
    class Partial_Update {
        private:
            std::vector<NN::Parameter<T>*> m_params;

        public: 
            Partial_Update() 
            {}

            void init(std::vector<NN::Parameter<T>*>& params) {
                m_params = params;
            }

            void step() {
                for(auto* param : m_params) {
                    param->value -= param->grad;
                    break;
                }
            }
    };

    template <std::floating_point T>
    class Biased_optimizer {
        private:
            std::vector<NN::Parameter<T>*> m_params;

        public: 
            Biased_optimizer() 
            {}

            void init(std::vector<NN::Parameter<T>*>& params) {
                m_params = params;
            }

            void step() {
                for(auto* param : m_params) {
                    param->value -= param->grad;
                    param->value += 0.01;
                }
            }
    };

    template <std::floating_point T>
    class Worthless {
        private:
            std::vector<NN::Parameter<T>*> m_params;

        public: 
            Worthless() 
            {}

            void init(std::vector<NN::Parameter<T>*>& params) {
                m_params = params;
            }

            void step() {
            }
    };
}

TEST(OptimizerCheckTest, PassesObviousOptimizer) {
    Obvious_optimizer<double> opt {};

    Debug::Optimizer_result result {Debug::optimizer_check<Obvious_optimizer<double>>(opt)};

    EXPECT_TRUE(result.passed);
}

TEST(OptimizerCheckTest, CatchesWrongDirectionUpdate) {
    Wrong_direction<double> opt {};

    Debug::Optimizer_result result {Debug::optimizer_check<Wrong_direction<double>>(opt)};

    EXPECT_TRUE(!result.descent_check && result.parameter_coverage && result.zero_gradient);
}

TEST(OptimizerCheckTest, CatchesPartialUpdate) {
    Partial_Update<double> opt {};

    Debug::Optimizer_result result {Debug::optimizer_check<Partial_Update<double>>(opt)};

    EXPECT_TRUE(result.descent_check && !result.parameter_coverage && result.zero_gradient);
}

TEST(OptimizerCheckTest, CathcesZeroGradientMover) {
    Biased_optimizer<double> opt {};

    Debug::Optimizer_result result {Debug::optimizer_check<Biased_optimizer<double>>(opt)};

    EXPECT_TRUE(result.parameter_coverage && !result.zero_gradient);
}

TEST(OptimizerCheckTest, CathesMultipleFaults) {
    Worthless<double> opt {};

    Debug::Optimizer_result result {Debug::optimizer_check<Worthless<double>>(opt)};

    EXPECT_TRUE(!result.descent_check && !result.parameter_coverage && result.zero_gradient);
}