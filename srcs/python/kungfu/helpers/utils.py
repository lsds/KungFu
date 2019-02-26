def show_size(s):
    Ki = 1024
    Mi = Ki * Ki
    Gi = Mi * Ki
    if s > Gi:
        return '%.2fGBytes' % (float(s) / Gi)
    elif s > Mi:
        return '%.2fMBytes' % (float(s) / Mi)
    elif s > Ki:
        return '%.2fKBytes' % (float(s) / Ki)
    else:
        return '%d' % s
