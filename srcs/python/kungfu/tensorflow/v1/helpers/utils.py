Ki = 1024
Mi = Ki * Ki
Gi = Mi * Ki


def show_size(s):
    if s > Gi:
        return '%.2fGi' % (float(s) / Gi)
    elif s > Mi:
        return '%.2fMi' % (float(s) / Mi)
    elif s > Ki:
        return '%.2fKi' % (float(s) / Ki)
    else:
        return '%d' % s


def show_rate(size, duration):
    r = size / duration
    if r < Ki:
        return '%.2fB/s' % r
    elif r < Mi:
        return '%.2fKiB/s' % (r / Ki)
    elif r < Gi:
        return '%.2fMiB/s' % (r / Mi)
    else:
        return '%.2fGiB/s' % (r / Gi)


def must_get_tensor_by_name(name):
    import tensorflow as tf
    realname = name + ':0'
    options = []
    for v in tf.global_variables():
        if v.name == realname:
            options.append(v)
    if len(options) < 1:
        raise RuntimeError('tensor named %s not found' % (name))
    if len(options) > 1:
        raise RuntimeError('more than one tensor named %s found' % (name))
    return options[0]
