import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re

import datetime

from collections import defaultdict
import math
import random

# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

matplotlib.rcParams["text.latex.preamble"].append(r'\mathchardef\mhyphen="2D')

class MyLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self, x, pos=None):
        # call the original LogFormatter
        rv = matplotlib.ticker.LogFormatterMathtext.__call__(self, x, pos)

        # check if we really use TeX
        if matplotlib.rcParams["text.usetex"]:
            # if we have the string ^{- there is a negative exponent
            # where the minus sign is replaced by the short hyphen
            rv = re.sub(r'\^\{-', r'^{\mhyphen', rv)

        return rv


def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def extract_from_worker(s, from_worker):
    pattern = re.compile(r"\[.*/0" + str(from_worker) + r"/.*\].*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return s

def print_none_match(pattern, l):
    print(pattern.pattern)
    raise Exception(l)

def get_accs(s):
    pattern = re.compile(r".*\)\t(?P<loss>\d+\.\d+)\t(?P<top1>\d+\.\d+)\t(?P<top5>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return (float(match.group("loss")), 1 -  float(match.group("top1")), 1- float(match.group("top5")))

def get_batch_and_noise(s, type_of_averaging):
    pattern = re.compile(r".*\[" + re.escape(type_of_averaging) + r"\]\sFuture\sbatch\s(?P<batch>\-?\d+\.\d+)\;\sNoise\s(?P<noise>\-?\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return (float(match.group("batch")), float(match.group("noise"))) 

def plot(ax, lines, type_of_averaging, worker, decay=-1):
    batches_and_noises = []
    for l in lines:
        pair = get_batch_and_noise(l, type_of_averaging)
        if pair is not None:
           batches_and_noises.append(pair)
    accs = []
    for l in lines:
        pair = get_accs(l)
        if pair is not None:
           accs.append(pair)

    losses, top1errors, top5errors = zip(*accs[len(accs) - len(batches_and_noises):])
    batches, noises = zip(*batches_and_noises)

    s = zip(batches_and_noises, top1errors)
    batches = []
    noises  = []
    top1errors   = []  
    for (b, n), acc in s:
        batches.append(b)
        noises.append(n)
        top1errors.append(acc)


    print(noises)

    ax.semilogy()
    ax.yaxis.set_major_formatter(MyLogFormatter())

    ax.semilogx()

    ax.plot(range(len(batches)), batches, color='b', zorder=5)
    ax.plot(range(len(noises)), noises, color='coral')

    ax.set_ylabel('Gradient Noise Scale')
    ax.set_xlabel('Iterations')

    if type_of_averaging == "EMA":
        ax.title.set_text('ResNet-32 Gradient Noise Scale Worker ' +  str(worker) +' (' + type_of_averaging + ' with Decay ' + str(decay) + ')')
    else:
        ax.title.set_text('ResNet-32 Gradient Noise Scale Worker ' +  str(worker) +' (' + type_of_averaging + ')')


def main():
    num_workers = 4
    workers = []
    for worker in range(num_workers):
        worker_running_sum = get_experiment_results('./decay-0.2-batch-0.2/noise-running-sum-decay-0.2-batch-0.2.log', lambda x: extract_from_worker(x, worker))
        worker_ema = get_experiment_results('./decay-0.2-batch-0.2/noise-ema-0.2-batch-0.2.log', lambda x: extract_from_worker(x, worker))
        workers.append((worker_running_sum, worker_ema))

    
    fig, axes = plt.subplots(4, 2, constrained_layout=False, sharex=True, sharey=True)

    index = 0
    axes_indices = [(i, j) for j in range(2) for i in range(4)]
    for worker, (worker_logs_running_sum, _) in enumerate(workers):
        i, j = axes_indices[index]
        plot(axes[i, j], worker_logs_running_sum, "Running Sum", worker)
        index += 1

    for worker, (_, worker_logs_ema) in enumerate(workers):
        i, j = axes_indices[index]
        plot(axes[i, j], worker_logs_ema, "EMA", worker, decay=0.2) ##### HERE
        index += 1

    fig.suptitle('ResNet-32 Gradient Noise Scale with Decay 0.2') ##### HERE

    plt.subplots_adjust(hspace=0.5)

    plt.show()

if __name__ == "__main__":
    main()
