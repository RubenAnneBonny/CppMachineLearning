#include <Rand/random.h>
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