#include <kungfu_base.h>

void invoke_callback(callback_t *f) { (*f)(); }

void delete_callback(callback_t *f) { delete f; }
