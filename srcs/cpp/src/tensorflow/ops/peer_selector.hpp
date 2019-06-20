#pragma once
#include <chrono>
#include <queue>
#include <vector>

class AdaptivePeerSelector
{
    using clock_t    = std::chrono::high_resolution_clock;
    using duration_t = std::chrono::duration<double>;

    struct weight {
        int count;
        double mean;

        weight() : weight(0, 0.0) {}

        weight(double value) : weight(1, value) {}

        weight(int count, double value) : count(count), mean(value) {}

        double total() const { return count * mean; }

        bool operator<(const weight &w) const
        {
            return count == 0 || mean < w.mean ||
                   (std::fabs(mean - w.mean) < 1e-6 && count < w.count);
        }

        void operator+=(const weight &w)
        {
            double tot = total() + w.total();
            count += w.count;
            if (count) {
                mean = tot / count;
            } else {
                mean = 0;
            }
        }

        weight operator-() const { return weight(-count, -mean); }
    };

    using Value = int;

    const std::vector<Value> values_;
    const std::map<Value, int> index_;
    const int window_size_;

    std::vector<weight> weights_;
    std::vector<weight> rolling_weights_;
    std::queue<weight> window_;

    static std::map<Value, int> BuildIndex(const std::vector<Value> &values)
    {
        std::map<Value, int> idx;
        const int n = values.size();
        for (int i = 0; i < n; ++i) { idx[values[i]] = i; }
        if (idx.size() != values.size()) {
            throw std::logic_error("duplicated values detected");
        }
        return idx;
    }

    int Next()
    {
        const int idx =
            std::min_element(rolling_weights_.begin(), rolling_weights_.end()) -
            rolling_weights_.begin();
        return values_.at(idx);
    }

    void Feedback(const Value &v, const duration_t &d)
    {
        const int idx = index_.at(v);
        const weight w(d.count());
        weights_.at(idx) += w;
        rolling_weights_.at(idx) += w;
        window_.push(w);
        if (window_.size() > window_size_) {
            const auto w = window_.front();
            rolling_weights_.at(idx) += -w;
        }
    }

  public:
    AdaptivePeerSelector(const std::vector<Value> &values, int window_size)
        : values_(values), index_(BuildIndex(values)), window_size_(window_size)
    {
        weights_.resize(values.size());
        rolling_weights_.resize(values.size());
        std::cout << "Using adaptive peer selector" << std::endl;
    }

    ~AdaptivePeerSelector() { ShowStat(); }

    void Do(const std::function<void(const Value &Value)> &task)
    {
        const Value v = Next();
        const auto t0 = clock_t::now();
        task(v);
        const duration_t d = clock_t::now() - t0;
        Feedback(v, d);
    }

    void ShowStat()
    {
        const int n = values_.size();
        for (int i = 0; i < n; ++i) {
            printf("%3d used %8d times, ", values_.at(i), weights_.at(i).count);
        }
        printf("\n");
    }
};
