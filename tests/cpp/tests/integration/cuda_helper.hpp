#pragma once
#include <string>

#include <cuda_runtime.h>

#include "error_checker.hpp"

struct show_cuda_error {
    std::string operator()(cudaError_t err) const
    {
        return cudaGetErrorString(err);
    }
};

using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;
