#pragma once
#include <memory>

#include <kungfu.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void kungfu_python_init();

#ifdef __cplusplus
}

extern std::unique_ptr<kungfu_world> _kungfu_world;

#endif
