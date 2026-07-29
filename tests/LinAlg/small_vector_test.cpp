#include <LinAlg/small_vector.h>
#include <gtest/gtest.h>

TEST(SmallVector, DefaultConstructor) {
    LinAlg::Small_vector<int, 4> sv;

    EXPECT_EQ(sv.size(), 0);
}

TEST(SmallVector, CountConstructor) {
    LinAlg::Small_vector<int, 4> sv(2);

    ASSERT_EQ(sv.size(), 2);
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[1], 0);
}

TEST(SmallVector, CountValueConstructor) {
    LinAlg::Small_vector<int, 4> sv(2, 5);

    ASSERT_EQ(sv.size(), 2);
    EXPECT_EQ(sv[0], 5);
    EXPECT_EQ(sv[1], 5);
}

TEST(SmallVector, InitializerListConstructor) {
    LinAlg::Small_vector<int, 4> sv {1, -4, 2};

    ASSERT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], -4);
    EXPECT_EQ(sv[2], 2);
}

TEST(SmallVector, VectorConstructor) {
    std::vector<int> v {3, 1};
    LinAlg::Small_vector<int, 6> sv {v};
    
    ASSERT_EQ(sv.size(), 2);
    EXPECT_EQ(sv[0], 3);
    EXPECT_EQ(sv[1], 1);
}

TEST(SmallVector, AccessOperator) {
    LinAlg::Small_vector<int, 4> sv {2, -6};

    sv[1] = 3;

    EXPECT_EQ(sv[0], 2);
    EXPECT_EQ(sv[1], 3);
}

TEST(SmallVector, Back) {
    LinAlg::Small_vector<int, 4> sv {2, -6};

    sv.back() = 3;

    EXPECT_EQ(sv[0], 2);
    EXPECT_EQ(sv[1], 3);
    EXPECT_EQ(sv.back(), 3);
}

TEST(SmallVector, PushBackWithinInlineCapacity) {
    LinAlg::Small_vector<int, 3> sv;

    sv.push_back(1);
    sv.push_back(2);
    sv.push_back(3);

    EXPECT_EQ(sv.capacity(), 3);
}

TEST(SmallVector, PuchBackGrowsToHeapPreservingData) {
    LinAlg::Small_vector<int, 2> sv;

    sv.push_back(1);
    sv.push_back(2);
    sv.push_back(3);

    EXPECT_EQ(sv.capacity(), 4);
}

TEST(SmallVector, PushBackManyPreservesOrder) {
    LinAlg::Small_vector<int, 4> sv;

    for(int i {}; i < 100; ++i) {
        sv.push_back(i);
    }

    ASSERT_EQ(sv.size(), 100);
    for(std::size_t i {}; i < 100; ++i) {
        EXPECT_EQ(sv[i], static_cast<int>(i));
    }
}

TEST(SmallVector, CopyConstructorIsDeppInline) {
    LinAlg::Small_vector<int, 4> sv {2, 1};
    LinAlg::Small_vector<int, 4> sv_copy {sv};

    sv_copy[0] = 8;

    EXPECT_EQ(sv[0], 2);
}

TEST(SmallVector, CopyConstructorIsDeepHeap) {
    LinAlg::Small_vector<int, 4> sv {2, 1, -5, 6, 12, 11};
    LinAlg::Small_vector<int, 4> sv_copy {sv};

    sv_copy[0] = 8;

    EXPECT_EQ(sv[0], 2);
}

TEST(SmallVector, CopyAssignmentIsDeep) {
    LinAlg::Small_vector<int, 4> sv {2, 1};
    LinAlg::Small_vector<int, 4> sv_copy;

    sv_copy = sv;

    sv_copy[0] = 8;

    EXPECT_EQ(sv[0], 2);
}

TEST(SmallVector, CopyAssignmentSelfIsSafe) {
    LinAlg::Small_vector<int, 4> sv {2, 1};

    sv = sv;

    sv[0] = 3;

    EXPECT_EQ(sv[0], 3);
    EXPECT_EQ(sv[1], 1);
}

TEST(SmallVector, MoveConstructorHeapAndResetSource) {
    LinAlg::Small_vector<int, 4> source;
    for(int i {}; i < 10; ++i) {
        source.push_back(i);
    }

    LinAlg::Small_vector<int, 4> moved {std::move(source)};

    ASSERT_EQ(moved.size(), 10);
    EXPECT_EQ(moved[9], 9);

    EXPECT_EQ(source.size(), 0);
    EXPECT_TRUE(source.empty());
    source.push_back(1);
    ASSERT_EQ(source.size(), 1);
    EXPECT_EQ(source[0], 1);
}

TEST(SmallVector, MoveConstructorInline) {
    LinAlg::Small_vector<int, 4> source {1, 2, 5};
    LinAlg::Small_vector<int, 4> moved {std::move(source)};

    ASSERT_EQ(moved.size(), 3);
    EXPECT_EQ(moved[0], 1);;
    EXPECT_EQ(moved[1], 2);
    EXPECT_EQ(moved[2], 5);

    EXPECT_EQ(source.size(), 3);
}

TEST(SmallVector, MoveAssignmentTransfersHeap) {
    LinAlg::Small_vector<int, 4> source;
    for(int i {}; i < 10; ++i) {
        source.push_back(i);
    }
    LinAlg::Small_vector<int, 4> target {4, 7};
    target = std::move(source);

    ASSERT_EQ(target.size(), 10);
    EXPECT_EQ(target[9], 9);
    EXPECT_EQ(source.size(), 0);
}

TEST(SmallVector, MoveAssignmentSelfIsSafe) {
    LinAlg::Small_vector<int, 4> sv;
    for(int i {}; i < 10; ++i) {
        sv.push_back(i);
    }
    LinAlg::Small_vector<int, 4>& sv_ref = sv;
    sv = std::move(sv_ref);

    ASSERT_EQ(sv.size(), 10);
    EXPECT_EQ(sv[9], 9);
}

TEST(SmallVector, AssignReplacesContents) {
    LinAlg::Small_vector<int, 4> sv {1, 2, 3};
    sv.assign(5, 8);

    ASSERT_EQ(sv.size(), 5);
    for(std::size_t i {}; i < 5; ++i) {
        EXPECT_EQ(sv[i], 8);
    }
}

TEST(SmallVector, ClearResetsSizeKeepsUsable) {
    LinAlg::Small_vector<int, 4> sv {1, 2, 3};
    sv.clear();

    EXPECT_EQ(sv.size(), 0);
    EXPECT_TRUE(sv.empty());

    sv.push_back(101);
    ASSERT_EQ(sv.size(), 1);
    EXPECT_EQ(sv[0], 101);
}

TEST(SmallVector, InsertSingleElement) {
    LinAlg::Small_vector<int, 4> sv {1, 2, 4};
    sv.insert(sv.begin() + 2, 3);

    ASSERT_EQ(sv.size(), 4);
    for(std::size_t i {}; i < 4; ++i) {
        EXPECT_EQ(sv[i], i + 1);
    }
}

TEST(SmallVector, InsertMultipleShiftsRight) {
    LinAlg::Small_vector<int, 8> sv {1, 4};
    sv.insert(sv.begin() + 1, 2, -9);

    ASSERT_EQ(sv.size(), 4);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], -9);
    EXPECT_EQ(sv[2], -9);
    EXPECT_EQ(sv[3], 4);
}

TEST(SmallVector, InsertCausingReallocation) {
    LinAlg::Small_vector<int, 4> sv {1, 2, 3};
    sv.insert(sv.begin() + 1, 2, 0);

    ASSERT_EQ(sv.size(), 5);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 0);
    EXPECT_EQ(sv[2], 0);
    EXPECT_EQ(sv[3], 2);
    EXPECT_EQ(sv[4], 3);
}

TEST(SmallVector, EraseRangeShiftsDown) {
    LinAlg::Small_vector<int, 8> sv {1, 2, 3, 4, 5};
    sv.erase(sv.begin() + 1, sv.begin() + 3);

    ASSERT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 5);
}

TEST(SmallVector, EraseSingleElement) {
    LinAlg::Small_vector<int, 8> sv {1, 2, 3};
    sv.erase(sv.begin() + 1);

    ASSERT_EQ(sv.size(), 2);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 3);
}

TEST(SmallVector, EqualityOperator) {
    LinAlg::Small_vector<int, 4> a {1, 2, 3};
    LinAlg::Small_vector<int, 4> b {1, 2, 3};
    LinAlg::Small_vector<int, 4> c {1, 2, 4};
    LinAlg::Small_vector<int, 4> d {1, 2};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
}

TEST(SmallVector, EqualityAcrossStorageModes) {
    LinAlg::Small_vector<int, 16> sv_inline;
    LinAlg::Small_vector<int, 4> sv_heap;
    for(int i {}; i < 8; ++i) {
        sv_inline.push_back(i);
        sv_heap.push_back(i);
    }

    ASSERT_EQ(sv_inline.size(), sv_heap.size());
    for(std::size_t i {}; i < sv_inline.size(); ++i) {
        EXPECT_EQ(sv_inline[i], sv_heap[i]);
    }
}

TEST(SmallVector, RangeForIteration) {
    LinAlg::Small_vector<int, 4> sv {1, -2, 3, 4, 5};
    int sum {};
    for(int x : sv) {
        sum += x;
    }
    EXPECT_EQ(sum, 11);
}

TEST(SmallVector, ConstAccessors) {
    const LinAlg::Small_vector<int, 4> sv {1, 2, 3};

    ASSERT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv.back(), 3);

    int sum {};
    for(const int* it {sv.begin()}; it != sv.end(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, 6);
}