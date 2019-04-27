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

class tensor_meta
{

  public:
    const std::string name;
    const int32_t size;

    tensor_meta(std::string &name, int32_t size) : name(name), size(size) {}

    friend std::ostream &operator<<(std::ostream &o, const tensor_meta &t)
    {
        o << "TensorMeta{name=" << t.name << ", size=" << t.size << "}"
          << std::endl;
        return o;
    }
};

// https://www.youtube.com/watch?v=uDdMuUWf6h4
class partition
{

  public:
    const int32_t index;
    const int32_t budget;
    int32_t current_cost;
    std::unordered_set<std::string> tensorNames;

    partition(int index, int budget)
        : index(index), budget(budget), current_cost(0)
    {
    }

    friend std::ostream &operator<<(std::ostream &os, const partition &p)
    {
        os << "Partition{budget=" << p.budget
           << ", current_cost=" << p.current_cost << "}" << std::endl;
        for (std::string tName : p.tensorNames) {
            os << "TensorMetaName{name=" << tName << "}" << std::endl;
        }
        return os;
    }

    bool put(const tensor_meta &t)
    {
        if (current_cost + t.size > budget) return false;

        tensorNames.insert(t.name);
        current_cost += t.size;

        return true;
    }
};

// Off-line bin packing algorithm
// 1. Best-fit decreasing NO
// 2. First-fit decreasing YES
class partial_exchange_manager
{
  public:
    int32_t budget;

    std::vector<tensor_meta *> tensors;
    std::vector<partition *> partitions;

    // Indicates the number of partitions
    int32_t bin_counter;

    partial_exchange_manager() : budget(0), countGradients(0), bin_counter(1) {}

    ~partial_exchange_manager()
    {
        for (tensor_meta *t_m : tensors) {
            delete t_m;
        }
        for (partition *p : partitions) {
            delete p;
        }
    }

    void setCountGradients(int32_t count)
    {
        std::lock_guard<std::mutex> lock(constructionMutex);
        if (this->countGradients == 0) {
            countGradients = count;
        }
    }

    void setBudget(int32_t budget)
    {
        std::lock_guard<std::mutex> lock(constructionMutex);
        if (this->budget == 0) {
            this->budget = budget;
        }
    }

    void addTensorInfo(std::string name, int32_t size)
    {
        std::lock_guard<std::mutex> lock(constructionMutex);
        tensor_meta *t_m = new tensor_meta(name, size);

        tensors.push_back(t_m);

        // Begin bin packing when all operators constructed
        if (tensors.size() == countGradients) {
            std::cout << "The budget is: " << this->budget << std::endl;
            bin_pack();
            print_partition_info();
        }
    }

    bool isReadyForNegotiation(std::string tensor_name, int32_t global_step)
    {
        // No need to lock because the partitioning is done.
        std::lock_guard<std::mutex> lock(partitionAccessMutex);
        auto partitionSet =
            partitions[global_step % partitions.size()]->tensorNames;
        return partitionSet.find(tensor_name) !=
               partitionSet.end();  // is present
    }

  private:
    int32_t countGradients;

    std::mutex constructionMutex;
    std::mutex partitionAccessMutex;

    void bin_pack()
    {
        std::cout << "Total budget per bin: " << budget << std::endl;
        std::cout << "When starting bin packing, the tensors are: "
                  << std::endl;
        for (tensor_meta *t : tensors) {
            std::cout << *t << std::endl;
        }

        // Comparator affects the ordering on each peer.
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

        for (tensor_meta *t : tensors) {
            bool currPartitionFilled   = false;
            int32_t currPartitionIndex = 0;

            while (!currPartitionFilled) {
                if (currPartitionIndex == partitions.size()) {
                    partition *newPartition =
                        new partition(++this->bin_counter, budget);
                    newPartition->put(*t);
                    partitions.push_back(newPartition);
                    currPartitionFilled = true;
                } else if (partitions[currPartitionIndex]->put(*t)) {
                    currPartitionFilled = true;
                } else {
                    currPartitionIndex++;  // move on to the next partition
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
};
