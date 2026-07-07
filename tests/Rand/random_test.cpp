#include <Rand/random.h>
#include <cassert>

void seeded_test(){
    Rand::Random<float> r_1 {42};
    Rand::Random<float> r_2 {42};

    assert(r_1.normal(0, 3) == r_2.normal(0, 3) && "The seeding does not work for normal distrobution");
    assert(r_1.uniform(0, 3) == r_2.uniform(0, 3) && "The seeding does not work for unifor distrobution");
}

void random_test(){
    Rand::Random<float> r_1 {};
    Rand::Random<float> r_2 {};
    Rand::Random<float> r_3 {};

    assert(r_1.normal(0, 3) != r_1.normal(0, 3) && "Unseeded randomizer gave same result twice");
    assert(r_2.normal(0, 3) != r_3.normal(0, 3) && "Two unseeded randomizer gave same result");
}

int main(){
    seeded_test();
    random_test();

    return 0;
}