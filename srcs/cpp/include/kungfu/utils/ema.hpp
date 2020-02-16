#pragma once
#include <type_traits>

namespace kungfu
{
template <typename T> class ExponentialMovingAverage
{
    T alpha_;
    bool has_value_;
    T value_;

  public:
    ExponentialMovingAverage(T alpha)
        : alpha_(alpha), has_value_(false), value_(0)
    {
        static_assert(std::is_floating_point<T>::value, "");
    }

    T update(T x)
    {
        if (has_value_) {
            value_ = alpha_ * value_ + (1 - alpha_) * x;
        } else {
            has_value_ = true;
            value_     = x;
        }
        return value_;
    }
};
}  // namespace kungfu
