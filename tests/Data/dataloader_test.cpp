#include <Rand/random.h>
#include <LinAlg/tensor.h>
#include <Data/dataloader.h>
#include <vector>
#include <gtest/gtest.h>

TEST(Dataloader, OneEpochDontLoseRows) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    for(int i {}; i < 100; ++i) {
        inputs[{i, 0}] = i;
    }

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader {random, inputs, targets, 100};

    LinAlg::Tensor<float> input {{1}};
    LinAlg::Tensor<float> target {{1}};

    loader.next_batch(random, input, target);

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

TEST(Dataloader, InputTargetMatch) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    for(int i {}; i < 100; ++i) {
        inputs[{i, 0}] = i;
        targets[{i, 0}] = i;
    }

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader {random, inputs, targets, 100};

    LinAlg::Tensor<float> input {{1}};
    LinAlg::Tensor<float> target {{1}};

    loader.next_batch(random, input, target);

    for(int i {}; i < 100; ++i) {
        float x {input[{i, 0}]};
        float y {target[{i, 0}]};

        EXPECT_EQ(x, y);
    }
}

TEST(Dataloader, NumBatchesNotDivisible) {
    LinAlg::Tensor<float> inputs {{42, 10}};
    LinAlg::Tensor<float> targets {{42, 2}};

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader {random, inputs, targets, 10};

    EXPECT_EQ(5, loader.get_num_batches());
}

TEST(Dataloader, Determinism) {
    std::vector<LinAlg::Tensor<float>> X {};

    for(int i {}; i < 2; ++i) {
        LinAlg::Tensor<float> inputs {{100, 1}};
        LinAlg::Tensor<float> targets {{100, 1}};

        Rand::Random<float> random {42};

        Data::Dataloader<float> loader {random, inputs, targets, 100};

        LinAlg::Tensor<float> input {{1}};
        X.push_back(input);
        LinAlg::Tensor<float> target {{1}};

        loader.next_batch(random, X[i], target);
    }

    EXPECT_EQ(X[0], X[1]);
}

TEST(Dataloader, TwoEpochsDiffer) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    for(int i {}; i < 100; ++i) {
        inputs[{i, 0}] = i;
        targets[{i, 0}] = i;
    }

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader {random, inputs, targets, 100};

    LinAlg::Tensor<float> input_1 {{1}};
    LinAlg::Tensor<float> input_2 {{1}};
    LinAlg::Tensor<float> target {{1}};

    loader.next_batch(random, input_1, target);
    loader.next_batch(random, input_2, target);
    loader.next_batch(random, input_2, target);

    EXPECT_NE(input_1, input_2);

    std::vector<int> found(100, 0);

    bool one_different {false};

    for(int i {}; i < 100; ++i) {
        EXPECT_FALSE(found[static_cast<int>(input_2[{i, 0}])]);

        if(static_cast<int>(input_2[{i, 0}]) != i) {
            one_different = true;
        }

        found[static_cast<int>(input_2[{i, 0}])] = true;
    }

    EXPECT_TRUE(one_different);

    for(int i {}; i < 100; ++i) {
        EXPECT_TRUE(found[i]);
    }
}

TEST(Dataloader, AfterLastBatchReturnsFalse) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    Rand::Random<float> random {42};

    Data::Dataloader<float> loader_1 {random, inputs, targets, 100};

    LinAlg::Tensor<float> input {{1}};
    LinAlg::Tensor<float> target {{1}};

    EXPECT_TRUE(loader_1.next_batch(random, input, target));
    EXPECT_FALSE(loader_1.next_batch(random, input, target));

    Data::Dataloader<float> loader_2 {random, inputs, targets, 60};

    EXPECT_TRUE(loader_2.next_batch(random, input, target));
    EXPECT_TRUE(loader_2.next_batch(random, input, target));
    EXPECT_FALSE(loader_2.next_batch(random, input, target));
}

TEST(Dataloader, BatchSizeLessThanOneThrows) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{100, 1}};

    Rand::Random<float> random {42};

    EXPECT_THROW((Data::Dataloader<float> {random, inputs, targets, 0}), std::invalid_argument);
    EXPECT_THROW((Data::Dataloader<float> {random, inputs, targets, -32}), std::invalid_argument);
}

TEST(Dataloader, BatchSizeDifferInputsTargetsThrows) {
    LinAlg::Tensor<float> inputs {{100, 1}};
    LinAlg::Tensor<float> targets {{24, 1}};

    Rand::Random<float> random {42};

    EXPECT_THROW((Data::Dataloader<float> {random, inputs, targets, 21}), std::invalid_argument);
}