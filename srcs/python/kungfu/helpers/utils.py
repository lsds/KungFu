def show_size(s):
    Ki = 1024
    Mi = Ki * Ki
    Gi = Mi * Ki
    if s > Gi:
        return '%.2fGi' % (float(s) / Gi)
    elif s > Mi:
        return '%.2fMi' % (float(s) / Mi)
    elif s > Ki:
        return '%.2fKi' % (float(s) / Ki)
    else:
        return '%d' % s
