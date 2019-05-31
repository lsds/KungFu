#pragma once
#include <stdint.h>
#include <iostream>
#include <string>
#include <unordered_set>

#include "tensor_meta.h"

class partition
{

  public:
    const int32_t index;
    const int32_t budget;
    int32_t current_cost;
    std::unordered_set<std::string> tensorNames;

    partition(int index, int budget) : index(index), budget(budget), current_cost(0) {}

    friend std::ostream &operator<<(std::ostream &os, const partition &p)
    {
        os << "Partition{budget=" << p.budget
           << ", current_cost=" << p.current_cost << "}" << std::endl;
        for (std::string tName : p.tensorNames) {
            os << "TensorMetaName{name=" << tName << "}" << std::endl;
        }
        return os;
    }

    bool put(const tensor_meta* t)
    {
        if (current_cost + t->size > budget) return false;

        tensorNames.insert(t->name);
        current_cost += t->size;

        return true;
    }
};
