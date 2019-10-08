#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// TODO: rename it to done_callback_t
typedef struct CallbackWrapper callback_t;

extern void invoke_callback(callback_t *);
extern void delete_callback(callback_t *);

typedef struct data_callback_s data_callback_t;
extern void invoke_data_callback(data_callback_t *, void *);
extern void delete_data_callback(data_callback_t *);

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

struct data_callback_s {
    using func_t = std::function<void(void *)>;

  public:
    explicit data_callback_s(const func_t &f) : f_(f) {}

    void operator()(void *data) { f_(data); }

  private:
    func_t f_;
};

#endif
