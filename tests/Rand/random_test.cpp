#include <Rand/random.h>
#include <cmath>
#include <gtest/gtest.h>

TEST(Random, Seeded) {
    Rand::Random<float> r_1 {42};
    Rand::Random<float> r_2 {42};

    EXPECT_EQ(r_1.normal(0, 3), r_2.normal(0, 3));
    EXPECT_EQ(r_1.uniform(0, 3), r_2.uniform(0, 3));
}

TEST(Random, Unseeded){
    Rand::Random<float> r_1 {};
    Rand::Random<float> r_2 {};
    Rand::Random<float> r_3 {};

    EXPECT_NE(r_1.normal(0, 3), r_1.normal(0, 3));
    EXPECT_NE(r_2.normal(0, 3), r_3.normal(0, 3));
}

TEST(Random, UniformWithinBounds) {
    Rand::Random<float> random {};

    float previous_element {-1};
    for(int i {}; i < 100; ++i) {
        float new_element {random.uniform(0, 1)};

        EXPECT_NE(previous_element, new_element);
        EXPECT_TRUE(new_element <= 1 && new_element >= 0);

        previous_element = new_element;
    }
}

TEST(Random, NormalRoughStats) {
    std::vector<float> elements {};
    Rand::Random<float> random {};

    for(int i {}; i < 10000; ++i) {
        elements.push_back(random.normal(3, 1));
    }

    float sum {};
    for(std::size_t i {}; i < 10000; ++i) {
        sum += elements[i];
    }

    float mean {sum / 10000};

    EXPECT_NEAR(mean, 3, 0.1);

    float var {};
    for(std::size_t i {}; i < 10000; ++i) {
        var += (elements[i] - mean) * (elements[i] - mean);
    }

    var /= 10000;

    float stddev {std::sqrt(var)};

    EXPECT_NEAR(stddev, 1, 0.1);
}

TEST(Random, DifferentSeedsDifferentValues) {
    Rand::Random<float> r_1 {42};
    Rand::Random<float> r_2 {43};

    EXPECT_NE(r_1.normal(0, 1), r_2.normal(0, 1));
}

TEST(Random, UniformHighLessThanLowThrows) {
    Rand::Random<float> random {42};

    EXPECT_THROW(random.uniform(3, 2.999f), std::invalid_argument);
}

TEST(Random, UniformIntHighLessOrEqualThanLowThrows) {
    Rand::Random<float> random {42};

    EXPECT_THROW(random.uniform_int(2, 2), std::invalid_argument);
    EXPECT_THROW(random.uniform_int(4, 3), std::invalid_argument);
}

TEST(Random, PermutationIsPermutation) {
    Rand::Random<float> random {42};

    std::vector<int> p {random.permutation(4)};
    std::vector<int> found(4, 0);

    for(std::size_t i {}; i < 4; ++i) {
        EXPECT_TRUE(p[i] >= 0 && p[i] < 4);

        if(p[i] >= 0 && p[i] < 4) {
            found[static_cast<std::size_t>(p[i])] = 1;
        }
    }

    for(std::size_t i {}; i < 4; ++i) {
        EXPECT_EQ(found[i], 1);
    }
}

TEST(Random, PermutationSeeded) {
    Rand::Random<float> r_1 {42};
    Rand::Random<float> r_2 {42};

    std::vector<int> p_1 {r_1.permutation(10)};
    std::vector<int> p_2 {r_2.permutation(10)};

    for(std::size_t i; i < 10; ++i) {
        EXPECT_EQ(p_1[i], p_2[i]);
    }
}

TEST(Random, PermutationZeroIsEmpty) {
    Rand::Random<float> random {42};

    std::vector<int> p {random.permutation(0)};

    EXPECT_TRUE(p.empty());
}

TEST(Random, PermutationLenghtLessThanOneThrows) {
    Rand::Random<float> random {42};

    EXPECT_THROW(random.permutation(-1), std::invalid_argument);
}

TEST(Random, UniformIntWithinBounds) {
    Rand::Random<float> random {42};

    for(int i {}; i < 1000; ++i) {
        int value {random.uniform_int(3, 7)};
        EXPECT_TRUE(value >= 3 && value < 7);
    }
}