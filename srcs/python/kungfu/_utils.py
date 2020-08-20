import os
import time


def map_maybe(f, lst):
    return [f(x) if x is not None else None for x in lst]


def measure(f):
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    return duration, result


def one_based_range(n):
    return range(1, 1 + n)


def show_duration(duration):
    if duration < 1:
        return '%.2fms' % (duration * 1e3)
    if duration < 60:
        return '%.2fs' % duration
    sec = int(duration)
    mm, ss = sec / 60, sec % 60
    if duration < 3600:
        return '%dm%ds' % (mm, ss)
    return '%dh%dm%ds' % (mm / 60, mm % 60, ss)


def _since_job_start():
    t0 = os.getenv('KUNGFU_JOB_START_TIMESTAMP') or '0'
    t0 = int(t0)
    return time.time() - t0


def _since_proc_start():
    t0 = os.getenv('KUNGFU_PROC_START_TIMESTAMP') or '0'
    t0 = int(t0)
    return time.time() - t0


def _log_event(name):
    t0 = os.getenv('KUNGFU_PROC_START_TIMESTAMP') or '0'
    t0 = int(t0)
    t1 = time.time()
    d = t1 - t0
    # d = _since_proc_start()
    print('TS=%f %s :: %s since proc started' % (t1, name, show_duration(d)))
