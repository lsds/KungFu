#pragma once
#ifdef __cplusplus
extern "C" {
#endif

extern void std_transform_2(const void *input_1, const void *input_2,
                            void *output, int n, int dtype, int binary_op);

typedef struct CallbackWrapper callback_t;

extern void invoke_callback(callback_t *);
extern void delete_callback(callback_t *);

extern void float16_sum(void *z, const void *x, const void *y, int len);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <functional>
struct CallbackWrapper {
    using func_t = std::function<void()>;

  public:
    explicit CallbackWrapper(const func_t &f) : f_(f) {}

    void operator()() { f_(); }

  private:
    func_t f_;
};
#endif
