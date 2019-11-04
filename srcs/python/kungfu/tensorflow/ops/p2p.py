from ._tf_oplib import _op_lib


def request_variable(target, version=None, name=None, shape=None, dtype=None):
    """
    target: a scalar tensor of int32
    version: a scalar tensor of int64
    name: string
    """
    if version is None:
        version = 0
        use_version = False
    else:
        use_version = True
    if name is None:
        raise RuntimeError('name is required')
    if shape is None:
        raise RuntimeError('shape is required')
    if dtype is None:
        raise RuntimeError('dtype is required')
    return _op_lib.kungfu_request_variable(target,
                                           version,
                                           tensor_name=name,
                                           shape=shape,
                                           T=dtype,
                                           use_version=use_version)


def request_variable_with_template(target, template, version=None):
    return request_variable(target,
                            version=version,
                            name=template.name,
                            shape=template.shape,
                            dtype=template.dtype)
