from .variables import (create_assign_op_for, create_placeholder_for,
                        create_setter, eval_batch_size,
                        eval_gradient_noise_scale, get_batch_size,
                        get_or_create_batch_size,
                        get_or_create_global_variable)

__all__ = [
    'create_assign_op_for',
    'create_placeholder_for',
    'create_setter',
    'eval_batch_size',
    'eval_gradient_noise_scale',
    'get_batch_size',
    'get_or_create_batch_size',
    'get_or_create_global_variable',
]
