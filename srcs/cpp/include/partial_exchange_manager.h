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
    std::unordered_map<int, float> schedule_;
    std::vector<int> repartition_steps_;
    std::vector<tensor_meta> tensors_;

    size_t count_gradients_;
    size_t total_size_bytes_;

    std::mutex mu_;

    void print_schedule() {
        std::cout << "Schedule" << std::endl;
        std::unordered_map<int, float>::iterator it;
        for ( it = schedule_.begin(); it != schedule_.end(); it++ )
        {
            std::cout << it->first  
                    << ':'
                    << it->second
                    << std::endl ;
        }
    }

    std::vector<partition> bin_pack(float budget)
    {   
        if((int) tensors_.size() != count_gradients_) {
            throw "Some tensors have not been added to the Partial Exchange Manager";
        }

        auto grt = [](tensor_meta& t1, tensor_meta& t2) {
            return t1.size > t2.size ||
                (t1.size == t2.size && t1.name > t2.name);
        };
        std::sort(tensors_.begin(), tensors_.end(), grt);

        if (tensors_[0].size > budget) {
            std::cout << "Infeasible to bin-pack: provide higher fraction." << std::endl;
            std::cout << "budget=" << budget << ", max tensor size=" << tensors_[0].size << std::endl;
            
            throw "Infeasible to bin-pack: provide higher fraction.";
        }

        //std::cout << "Partitioning budget is " << budget << std::endl;

        int bin_counter = 1;
        std::vector<partition> partitions;
        partitions.push_back(partition(bin_counter, budget));

        for (const tensor_meta t : tensors_) {
            bool currPartitionFilled   = false;
            int32_t currPartitionIndex = 0;

            while (!currPartitionFilled) {
                if (currPartitionIndex == (int) partitions.size()) {
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

    void addSchedule(std::vector<int>& steps, std::vector<float>& fractions) {
        std::lock_guard<std::mutex> l(mu_);
        if(steps.size() != fractions.size()) {
            std::cout << "Invalid schedule: number of steps must equal number of fractions" << std::endl;
            throw "Invalid schedule: number of steps must equal number of fractions";
        }

        repartition_steps_ = std::vector<int>(steps.size(), UINT32_MAX);
        for(size_t i = 0; i < steps.size(); i++) {
            schedule_[steps[i]] = fractions[i];
            repartition_steps_[i] = steps[i];
        }
    }

    void addGlobalTensorInfo(size_t count_gradients, size_t total_size_bytes) {
        std::lock_guard<std::mutex> l(mu_);
        count_gradients_  = count_gradients;
        total_size_bytes_ = total_size_bytes; 
    }

    void addTensorInfo(std::string name, const int32_t size)
    {
        std::lock_guard<std::mutex> l(mu_);
        tensor_meta t_m(name, size);

        tensors_.push_back(t_m);
    }


    Plan repartition(int64_t gs) {
        std::lock_guard<std::mutex> l(mu_);        

        size_t next_step = UINT32_MAX;
        for(size_t i = 0; i < repartition_steps_.size() - 1; i++)  {
            if (gs == repartition_steps_[i]) { 
                next_step = repartition_steps_[i + 1]; 
                break;
            }
        }

        
        float new_fraction = -1;
        if (schedule_.find(gs) == schedule_.end()) {
            std::cout << "gs " << gs << " not in schedule map. " << std::endl;
            throw;
        } else {
            new_fraction = schedule_[gs];
        }

        print_schedule();

        if (new_fraction == -1) {
            return Plan(next_step, std::vector<partition>());
        }

        std::vector<partition> new_partitions = bin_pack(new_fraction * total_size_bytes_);

        return Plan(next_step, new_partitions);
    }
};
