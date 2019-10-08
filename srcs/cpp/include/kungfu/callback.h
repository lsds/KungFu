#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// TODO: rename it to done_callback_t
typedef struct CallbackWrapper callback_t;

extern void invoke_callback(callback_t *);
extern void delete_callback(callback_t *);

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
