from ._tf_oplib import _op_lib


def save_variable(t, version=None, name=None):
    """
    t: the tensor variable to save
    version: a scalar tensor of int64 or None
    """
    if version is None:
        version = 0
        use_version = False
    else:
        use_version = True
    if name is None:
        name = t.name
    return _op_lib.kungfu_save_variable(version,
                                        t,
                                        input_tensor_name=name,
                                        use_version=use_version)


def save_variables(variables):
    return _op_lib.kungfu_save_variables(variables,
                                         names=[v.name for v in variables])


def save_model(variables):
    import tensorflow as tf
    var_sizes = [var.shape.num_elements()
                 for var in variables]  # number of floats it has
    return _op_lib.save_model(variables,
                              var_type_size=variables[0].dtype.size,
                              var_sizes=var_sizes)
