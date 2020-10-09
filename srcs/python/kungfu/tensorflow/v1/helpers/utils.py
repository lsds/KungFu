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
