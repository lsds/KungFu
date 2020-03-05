#!/usr/bin/env python3


def gen_config():
    stage_sizes = [1, 2, 4, 8]
    step_per_stage = 3

    config = ','.join('%d:%d' % (size, step_per_stage) for size in stage_sizes)
    max_step = step_per_stage * len(stage_sizes)
    return config, max_step


config, max_step = gen_config()

print('--schedule %s' % (config))
print('--max-step %s' % (max_step))
