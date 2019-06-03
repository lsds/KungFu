#include "testing.hpp"

TEST(kungfu_test, test_type_size)
{
    ASSERT_EQ(kungfu_type_size(KungFu_INT32), static_cast<uint32_t>(4));
    ASSERT_EQ(kungfu_type_size(KungFu_FLOAT16), static_cast<uint32_t>(2));
    ASSERT_EQ(kungfu_type_size(KungFu_FLOAT), static_cast<uint32_t>(4));
    ASSERT_EQ(kungfu_type_size(KungFu_DOUBLE), static_cast<uint32_t>(8));
}

TEST(kungfu_test, test_transform)
{
    int n = 1;
    std::vector<float> x(n);
    std::vector<float> y(n);
    x[0] = 1;
    y[0] = 2;
    std_transform_2(x.data(), y.data(), x.data(), n, KungFu_FLOAT, KungFu_SUM);
    ASSERT_FLOAT_EQ(x[0], static_cast<float>(3));
}

TEST(kungfu_test, test_model_averaging) {

    float input[] = {1.5, 2.5, 100.0};
    float other[] = {1.5, 3.5, 50.0};

    float output[] {0.0, 0.0};

    float* input_ptr = &input[0];
    float* other_ptr = &other[0];
    float* output_ptr = &output[0];
    for (int j = 0; j < 3; j++) {
        *output_ptr = 0.5 * (*input_ptr + *other_ptr);
        output_ptr += 1;
        input_ptr += 1;
        other_ptr += 1;
    }

    ASSERT_FLOAT_EQ(output[0], static_cast<float>(1.5));
    ASSERT_FLOAT_EQ(output[1], static_cast<float>(3));
    ASSERT_FLOAT_EQ(output[2], static_cast<float>(75));


}
