#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace stdml
{
class ElasticState
{
    int job_start_;
    int proc_start_;

    int64_t progress_;
    int rank_;
    int size_;

    friend bool parse_elastic_state(ElasticState &e);

  public:
    ElasticState();

    std::string str() const;

    int64_t progress() const
    {
        return progress_;
    }

    int rank() const
    {
        return rank_;
    }

    int size() const
    {
        return size_;
    }
};

bool parse_elastic_state(ElasticState &e);

void gen_tf_record();
}  // namespace stdml
