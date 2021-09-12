#pragma once
#include <string>

#include <stdml/tensor>

namespace stdml::data
{
void decode_jpeg(const std::string &s);
void decode_resize_jpeg_hw3(const std::string &s, TensorRef x);
Tensor decode_resize_jpeg_hw3(const std::string &s, int h, int w);
}  // namespace stdml::data
