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


/* Off-line bin packing algorithm
   1. Best-fit decreasing NO
   2. First-fit decreasing YES
   Bin packing algorithm based on:
   https://www.youtube.com/watch?v=uDdMuUWf6h4 
*/
class partial_exchange_manager
{
    int32_t countGradients;

    std::mutex constructionMutex;
    std::mutex partitionAccessMutex;

    int32_t total_size_gradients;
    std::unordered_set<int> repartition_ids;

  
    void bin_pack()
    {   
        auto grt = [](tensor_meta *t1, tensor_meta *t2) {
            return t1->size > t2->size ||
                (t1->size == t2->size && t1->name > t2->name);
        };
        std::sort(tensors.begin(), tensors.end(), grt);

        if (tensors[0]->size > budget) {
            std::cout << "Infeasible to bin-pack the tensors with small "
                        "budget. Provide higher fraction."
                    << std::endl;
            throw "Infeasible to bin-pack the tensors with small budget. "
                "Provide higher fraction.";
            return;
        }

        partitions.push_back(new partition(this->bin_counter, budget));

        for (const tensor_meta* t : tensors) {
            bool currPartitionFilled   = false;
            int32_t currPartitionIndex = 0;

            while (!currPartitionFilled) {
                if (currPartitionIndex == partitions.size()) {
                    partition* newPartition = new partition(++this->bin_counter, budget);
                    newPartition->put(t);
                    partitions.push_back(newPartition);
                    currPartitionFilled = true;
                } else if (partitions[currPartitionIndex]->put(t)) {
                    currPartitionFilled = true;
                } else {
                    currPartitionIndex++;
                }
            }
        }
    }

    void print_partition_info()
    {
        std::cout << "Initial total number of tensors: " << countGradients
                  << std::endl;
        int32_t countTensorsFromParts = 0;
        std::cout << "Total number of partitions: " << bin_counter << std::endl;
        for (int i = 0; i < bin_counter; i++) {
            std::cout << "Partition: " << std::endl;
            std::cout << *partitions[i] << std::endl;
            countTensorsFromParts += partitions[i]->tensorNames.size();
        }
        std::cout << "Total number of tensors in partitions: "
                  << countTensorsFromParts << std::endl;
        if (countGradients != countTensorsFromParts) {
            throw "Bin packing error: tensor names are not unique.";
        }
    }

  public:
    int32_t budget;

    std::vector<tensor_meta *> tensors;
    std::vector<partition *> partitions;

    // Indicates the number of partitions
    int32_t bin_counter;
    float current_fraction;

    partial_exchange_manager() : 
            budget(0), countGradients(0), bin_counter(1), 
            current_fraction(0), total_size_gradients(0) {}

    ~partial_exchange_manager()
    {
        
    }

    void addTensorInfo(std::string name, const int32_t size)
    {
        std::lock_guard<std::mutex> lock(constructionMutex);
        tensor_meta *t_m = new tensor_meta(name, size);

        tensors.push_back(t_m);
    }

    bool isReadyForNegotiation(const std::string tensor_name, int32_t global_step)
    {
        std::lock_guard<std::mutex> lock(partitionAccessMutex);
        auto partitionSet = partitions[global_step % partitions.size()]->tensorNames;
        return partitionSet.find(tensor_name) != partitionSet.end();
    }


    void repartition(float new_fraction, int repartition_id) {
        std::lock_guard<std::mutex> partitionAccessLock(partitionAccessMutex);
        std::lock_guard<std::mutex> constructionLock(constructionMutex);

        if(repartition_ids.find(repartition_id) != repartition_ids.end()) {
            // present
            return;
        }

        repartition_ids.insert(repartition_id);
        
        
        this->budget = new_fraction * this->total_size_gradients;
        this->current_fraction = new_fraction;

        partitions.clear();

        // Restore bin counter
        this->bin_counter = 1;
        bin_pack();
    }
};
