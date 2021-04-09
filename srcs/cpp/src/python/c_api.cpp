#include <kungfu.h>
#include <kungfu/python/c_api.h>

void kungfu_all_reduce_int_max(int *px)
{
    int32_t x, y;
    x = *px;
    _default_peer->AllReduce(&x, &y, 1, KungFu_INT32, KungFu_MAX, "");
    *px = y;
}

void kungfu_resize(int n, char *p_changed, char *p_detached)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    _default_peer->ResizeCluster(n, reinterpret_cast<bool *>(p_changed),
                                 reinterpret_cast<bool *>(p_detached));
}

void kungfu_resize_from_url(char *p_changed, char *p_detached)
{
    static_assert(sizeof(bool) == sizeof(char), "");
    _default_peer->ResizeClusterFromURL(reinterpret_cast<bool *>(p_changed),
                                        reinterpret_cast<bool *>(p_detached));
}
