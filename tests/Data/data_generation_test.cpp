#include <cml.h>
#include <gtest/gtest.h>

TEST(DataGeneration, MakeCirclesShape) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 20)};

    EXPECT_EQ(data.inputs.get_rank(), 2);
    EXPECT_EQ(data.inputs.get_extent(0), 20);
    EXPECT_EQ(data.inputs.get_extent(1), 2);
    EXPECT_EQ(data.targets.get_extent(0), 20);
    EXPECT_EQ(data.targets.get_extent(1), 2);
}

TEST(DataGeneration, MakeMoonsShape) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_moons<double>(random, 20)};

    EXPECT_EQ(data.inputs.get_rank(), 2);
    EXPECT_EQ(data.inputs.get_extent(0), 20);
    EXPECT_EQ(data.inputs.get_extent(1), 2);
    EXPECT_EQ(data.targets.get_extent(0), 20);
    EXPECT_EQ(data.targets.get_extent(1), 2);
}

TEST(DataGeneration, MakeSpiralsShape) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 20)};

    EXPECT_EQ(data.inputs.get_rank(), 2);
    EXPECT_EQ(data.inputs.get_extent(0), 60);
    EXPECT_EQ(data.inputs.get_extent(1), 2);
    EXPECT_EQ(data.targets.get_extent(0), 60);
    EXPECT_EQ(data.targets.get_extent(1), 3);
}

TEST(DataGeneration, MakeBlackScholesShape) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_black_scholes<double>(random, 20)};

    EXPECT_EQ(data.inputs.get_rank(), 2);
    EXPECT_EQ(data.inputs.get_extent(0), 20);
    EXPECT_EQ(data.inputs.get_extent(1), 4);
    EXPECT_EQ(data.targets.get_extent(0), 20);
    EXPECT_EQ(data.targets.get_extent(1), 1);
}

TEST(DataGeneration, MakeCirclesOneHot) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_circles<double>(random, 1)};

    int ones {};
    int zeros {};

    for(int i {}; i < 2; ++i) {
        if(data.targets[{0, i}] == 0) {
            ++zeros;
        }
        else if(data.targets[{0, i}] == 1) {
            ++ones;
        }
    }    

    EXPECT_EQ(zeros, 1);
    EXPECT_EQ(ones, 1);
}

TEST(DataGeneration, MakeMoonsOneHot) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_moons<double>(random, 1)};

    int ones {};
    int zeros {};

    for(int i {}; i < 2; ++i) {
        if(data.targets[{0, i}] == 0) {
            ++zeros;
        }
        else if(data.targets[{0, i}] == 1) {
            ++ones;
        }
    }    

    EXPECT_EQ(zeros, 1);
    EXPECT_EQ(ones, 1);
}

TEST(DataGeneration, MakeSpiralsOneHot) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 1)};

    for(int j {}; j < 3; ++j) {
        int ones {};
        int zeros {};

        for(int i {}; i < 3; ++i) {
            if(data.targets[{j, i}] == 0) {
                ++zeros;
            }
            else if(data.targets[{j, i}] == 1) {
                ++ones;
            }
        }    

        EXPECT_EQ(zeros, 2);
        EXPECT_EQ(ones, 1);
    }
}

TEST(DataGeneration, MakeSpiralsAllClasses) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 1)};

    std::vector<int> found(3, 0);

    for(int i {}; i < 3; ++i) {
        for(int j {}; j < 3; ++j) {
            if(data.targets[{i, j}] == 1) {
                found[static_cast<std::size_t>(i)] = 1;
            }
        }
    }

    for(std::size_t i {}; i < 3; ++i) {
        EXPECT_EQ(found[i], 1);
    }
}

TEST(DataGeneration, BlackScholesCall) {
    double call_price {Data::black_scholes_call<double>(1, 1, 1, 0.2, 0)};

    EXPECT_NEAR(call_price, 0.0796557, 1e-6);
}

TEST(DataGeneration, BlackScholesLessThanOne) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_black_scholes<double>(random, 20)};

    for(int i {}; i < 20; ++i) {
        EXPECT_TRUE((data.targets[{i, 0}] >= 0 && data.targets[{i, 0}] <= 1));
    }
}

TEST(DataGeneration, MakeSpiralsShuffle) {
    Rand::Random<double> random {42};
    Data::Data_set<double> data {Data::make_spirals<double>(random, 200)};

    int times {};

    int index {};

    for(int i {}; i < 3; ++i) {
        if(data.targets[{0, i}] == 1) {
            index = i;
        }
    }

    for(int i {}; i < 60; ++i) {
        if(data.targets[{i, index}] == 1) {
            times++;
        }
    }

    EXPECT_NE(times, 60);
}

TEST(DataGeneration, DataSetShuffleSameShape) {
    LinAlg::Tensor<float> A {{10, 1}};
    LinAlg::Tensor<float> B {{10, 1}};

    for(int i {}; i < 10; ++i) {
        A[{i, 0}] = static_cast<float>(i);
        B[{i, 0}] = static_cast<float>(i);
    }

    Rand::Random<float> random {42};
    Data::Data_set<float> data {A, B};
    data.shuffle(random);

    ASSERT_EQ(data.inputs.get_rank(), 2);
    ASSERT_EQ(data.targets.get_rank(), 2);
    EXPECT_EQ(data.inputs.get_extent(0), 10);
    EXPECT_EQ(data.inputs.get_extent(1), 1);
    EXPECT_EQ(data.targets.get_extent(0), 10);
    EXPECT_EQ(data.targets.get_extent(1), 1);
}

TEST(DataGeneration, DataSetShufflePairs) {
    LinAlg::Tensor<float> A {{10, 1}};
    LinAlg::Tensor<float> B {{10, 1}};

    for(int i {}; i < 10; ++i) {
        A[{i, 0}] = static_cast<float>(i);
        B[{i, 0}] = static_cast<float>(i);
    }

    Rand::Random<float> random {42};
    Data::Data_set<float> data {A, B};
    data.shuffle(random);

    for(int i {}; i < 10; ++i) {
        EXPECT_EQ((data.inputs[{i, 0}]), (data.targets[{i, 0}]));
    }
}

TEST(DataGeneration, DataSetDifferentRankThrows) {
    LinAlg::Tensor<double> X {{4, 2}};
    LinAlg::Tensor<double> Y {{4}};

    EXPECT_THROW((Data::Data_set<double>{X, Y}), std::invalid_argument);
}

TEST(DataGeneration, DataSetDifferentBatchThrows) {
    LinAlg::Tensor<double> X {{4, 2}};
    LinAlg::Tensor<double> Y {{3, 2}};

    EXPECT_THROW((Data::Data_set<double>{X, Y}), std::invalid_argument);
}