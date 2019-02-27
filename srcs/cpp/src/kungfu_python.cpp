#include <kungfu_python.h>

std::unique_ptr<kungfu_world> _kungfu_world;

void kungfu_python_init() { _kungfu_world.reset(new kungfu_world); }
