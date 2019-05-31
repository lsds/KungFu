#pragma once
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


#include "partition.h"

/* Off-line bin packing algorithm
   1. Best-fit decreasing NO
   2. First-fit decreasing YES
   Bin packing algorithm based on:
   https://www.youtube.com/watch?v=uDdMuUWf6h4 
*/
class partial_exchange_manager
{
    std::vector<std::pair<int, int>> schedule_;
    std::vector<tensor_meta> tensors_;

    int32_t count_gradients_;
    int32_t total_size_bytes_;

    std::mutex mu_;


    std::vector<partition> bin_pack(int32_t budget)
    {   
        if(tensors_.size() != count_gradients_) {
            throw "Some tensors have not been added to the Partial Exchange Manager";
        }

        auto grt = [](tensor_meta t1, tensor_meta t2) {
            return t1.size > t2.size ||
                (t1.size == t2.size && t1.name > t2.name);
        };
        std::sort(tensors_.begin(), tensors_.end(), grt);

        if (tensors_[0].size > budget) {
            throw "Infeasible to bin-pack: provide higher fraction.";
        }

        int bin_counter = 1;
        std::vector<partition> partitions;
        partitions.push_back(partition(bin_counter, budget));

        for (const tensor_meta t : tensors_) {
            bool currPartitionFilled   = false;
            int32_t currPartitionIndex = 0;

            while (!currPartitionFilled) {
                if (currPartitionIndex == partitions.size()) {
                    partition newPartition = partition(++bin_counter, budget);
                    newPartition.put(t);
                    partitions.push_back(newPartition);
                    currPartitionFilled = true;
                } else if (partitions[currPartitionIndex].put(t)) {
                    currPartitionFilled = true;
                } else {
                    currPartitionIndex++;
                }
            }
        }
        return partitions;
    }

  public:

    partial_exchange_manager() : count_gradients_(0), total_size_bytes_(0) {}

    ~partial_exchange_manager() {}

    void addSchedule(std::vector<int> steps, std::vector<int> fractions) {
        std::lock_guard<std::mutex> l(mu_);
        if(steps.size() != fractions.size()) {
            throw "Invalid schedule: number of steps must equal number of fractions";
        }
        for(int i = 0; i < steps.size(); i++) {
            schedule_.push_back({steps[i], fractions[i]});
        }
    }

    void addGlobalTensorInfo(int count_gradients, int total_size_bytes) {
        std::lock_guard<std::mutex> l(mu_);
        count_gradients_  = count_gradients;
        total_size_bytes_ = total_size_bytes_; 
    }

    void addTensorInfo(std::string name, const int32_t size)
    {
        std::lock_guard<std::mutex> l(mu_);
        tensor_meta t_m(name, size);

        tensors_.push_back(t_m);
    }


    void repartition(int gs) {
        std::lock_guard<std::mutex> l(mu_);        


        int new_fraction = -1;
        int next_repartitioning_step = -1;
        for(int i = 0; i < schedule_.size(); i++) {
            int schedule_gs   = schedule_[i][0];
            int schedule_frac = schedule_[i][1];
            if(schedule_gs == gs) {
                new_fraction = schedule_frac;
            } 
        }

        if (new_fraction == -1) {
            return Plan(-1, );
        }

        std::vector<partition> newPartitions = bin_pack(new_fraction * total_size_bytes_);

        return Plan(next_repart_step, newPartitions);
    }
};
