#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct pool_s pool_t;

extern pool_t *new_pool();
extern void del_pool(pool_t *p);
extern void *get_buffer(pool_t *p, int size);
extern void put_buffer(pool_t *p, void *ptr);

#ifdef __cplusplus
}
#endif
