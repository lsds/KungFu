#pragma once
#include <cinttypes>

std::string to_hex(uint32_t x)
{
    char line[128];
    sprintf(line, "0x%08" PRIx32, x);
    return line;
}
