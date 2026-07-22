#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Data/dataloader.h>
#include <vector>
#include <gtest/gtest.h>

TEST(Dataloader, TruePermutation) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    for(int i {}; i < 100; ++i) {
        inputs[{i, 0}] = i;
    }

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader {random, inputs, targets, 100};

    LinAlg::Tensor<float> input {{1}};
    LinAlg::Tensor<float> target {{1}};

    loader.next_batch(input, target);

    std::vector<int> found(100, 0);

    bool one_different {false};

    for(int i {}; i < 100; ++i) {
        EXPECT_FALSE(found[static_cast<int>(input[{i, 0}])]);

        if(static_cast<int>(input[{i, 0}]) != i) {
            one_different = true;
        }

        found[static_cast<int>(input[{i, 0}])] = true;
    }

    EXPECT_TRUE(one_different);

    for(int i {}; i < 100; ++i) {
        EXPECT_TRUE(found[i]);
    }
}