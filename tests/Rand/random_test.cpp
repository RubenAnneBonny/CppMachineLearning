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
    for(int i {}; i < 10000; ++i) {
        sum += elements[i];
    }

    float mean {sum / 10000};

    EXPECT_NEAR(mean, 3, 0.1);

    float var {};
    for(int i {}; i < 10000; ++i) {
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