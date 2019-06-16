#pragma once
#include <chrono>
#include <vector>

class AdaptivePeerSelector
{
    using clock_t    = std::chrono::high_resolution_clock;
    using duration_t = std::chrono::duration<double>;

    struct weight {
        int count;
        double mean;

        weight() : count(0), mean(0.0) {}

        bool operator<(const weight &w) const
        {
            return count == 0 || mean < w.mean ||
                   (std::fabs(mean - w.mean) < 1e-6 && count < w.count);
        }

        void add(double delta)
        {
            double tot = mean * count + delta;
            ++count;
            mean = tot / count;
        }
    };

    using Value = int;

    const std::vector<Value> values_;
    const std::map<Value, int> index_;
    std::vector<weight> weights_;

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
        const int idx = std::min_element(weights_.begin(), weights_.end()) -
                        weights_.begin();
        return values_.at(idx);
    }

    void Feedback(const Value &v, const duration_t &d)
    {
        const int idx = index_.at(v);
        weights_.at(idx).add(d.count());
    }

  public:
    AdaptivePeerSelector(const std::vector<Value> &values)
        : values_(values), index_(BuildIndex(values))
    {
        weights_.resize(values.size());
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
