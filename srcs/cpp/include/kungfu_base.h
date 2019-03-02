#pragma once
#ifdef __cplusplus
extern "C" {
#endif

extern void std_transform_2(const void *input_1, const void *input_2,
                            void *output, int n, int dtype, int binary_op);

typedef struct CallbackWrapper callback_t;

extern void invoke_callback(callback_t *);
extern void delete_callback(callback_t *);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <functional>
#include <iostream>
struct CallbackWrapper {
    using func_t = std::function<void()>;

  public:
    explicit CallbackWrapper(const func_t &f) : f_(f) {}

    void operator()() { 
      std::cout << "Before calling the callback [WRAPPER]" << std::endl;
      f_();   
      std::cout << "After calling the callback [WRAPPER]" << std::endl;
    }

  private:
    func_t f_;
};
#endif
