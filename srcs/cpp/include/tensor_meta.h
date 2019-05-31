#pragma once
#include <stdint.h>
#include <iostream>
#include <string>

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