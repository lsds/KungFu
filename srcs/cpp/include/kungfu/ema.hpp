#pragma once

template <typename T> class ExponentialMovingAverage
{
    const T alpha_;

    int count_;
    T value_;

  public:
    ExponentialMovingAverage(const T alpha)
        : alpha_(alpha), count_(0), value_(0)
    {
    }

    void update(const T &x)
    {
        if (count_ == 0) {
            value_ = x;
        } else {
            value_ = alpha_ * x + (1 - alpha_) * value_;
        }
        ++count_;
    }

    T get() const { return value_; }
};