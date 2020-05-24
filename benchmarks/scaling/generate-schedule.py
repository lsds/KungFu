#!/usr/bin/env python3


def gen_config(period=5, cap=4):
    schedule = []
    for i in range(100):
        n = [1, cap][i % 2]
        schedule.append('%d:%d' % (i * period, n))
    return ','.join(schedule)


print(gen_config())
