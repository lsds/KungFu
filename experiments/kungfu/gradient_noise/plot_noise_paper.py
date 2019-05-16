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
from matplotlib.pyplot import figure


# Set size of figure
figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')


# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

font = {'family' : 'normal',
        'weight' : '300',
        'size'   : 11}

matplotlib.rc('font', **font)


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

def get_accs_mnist(s):
    pattern = re.compile(r".*Training\saccuracy\:\s(?P<trainacc>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return (-1, 1 - float(match.group("trainacc")), -1)


def get_accs_resnet32(s):
    pattern = re.compile(r".*\)\t(?P<loss>\d+\.\d+)\t(?P<top1>\d+\.\d+)\t(?P<top5>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    print("Matches")
    return (float(match.group("loss")), 1 -  float(match.group("top1")), 1- float(match.group("top5")))


def get_batch_and_noise(s, type_of_averaging):
    pattern = re.compile(r".*\[" + re.escape(type_of_averaging) + r"\]\sFuture\sbatch\s(?P<batch>\-?\d+\.\d+)\;\sNoise\s(?P<noise>\-?\d+[\.\d+]?).*", re.VERBOSE)
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

    # ## With accuracies
    # accs = []
    # for l in lines:
    #     pair = get_accs_mnist(l)
    #     if pair is not None:
    #        accs.append(pair)


    # # With warmup
    # # losses, top1errors, top5errors = zip(*accs[len(accs) - len(batches_and_noises):])
    # losses, top1errors, top5errors = zip(*accs)
    # batches, noises = zip(*batches_and_noises)

    # # s = zip(batches_and_noises, top1errors)
    # # s.sort(key=lambda x: x[1])
    # #s.reverse()
    # batches = []
    # noises  = []
    # top1errors   = []  
    # for (b, n), acc in s:
    #     batches.append(b)
    #     noises.append(n)
    #     top1errors.append(acc)

    # Without accuracies
    batches, noises = zip(*batches_and_noises)

    batches = batches[:1200]
    noises  = noises[:1200]

    ax.semilogy()
    ax.yaxis.set_major_formatter(MyLogFormatter())  
    ax.semilogx()


    ax_twin = ax.twinx()
    ax_twin.semilogy()
    ax_twin.yaxis.set_major_formatter(MyLogFormatter())


    ## Accuracies on the x axis
    #ax.plot(top1errors, batches, color='b', zorder=5, label="Estimated Batch Size")
    #ax.plot(top1errors, noises, color='coral', label="Gradient Noise")

    # Iterations on the x axis
    ax.plot(range(len(batches)), batches, color='b', linestyle='--', zorder=5, label="Estimated Batch Size")
    ax.plot(range(len(batches)), noises, color='coral', label="Gradient Noise")
    ax_twin.plot(range(len(batches)), batches, color='b', linestyle='--', zorder=5, label="Estimated Batch Size")
    ax_twin.plot(range(len(batches)), noises, color='coral', label="Gradient Noise")



    ax.set_ylabel('Gradient Noise Scale')
    ax.set_xlabel('Iterations')


    ax.legend()

    if type_of_averaging == "EMA":
        ax.title.set_text('LeNet-5 Gradient Noise Scale Worker ' +  str(worker) +' (' + type_of_averaging + ' with Decay ' + str(decay) + ')')
    else:
        type_of_averaging = "Running Average"
        #ax.title.set_text('LeNet-5 Gradient Noise Scale Worker ' +  str(worker) +' (' + type_of_averaging + ')')

def paper_figure():
    num_workers = 1
    workers = []
    for worker in [3]:
        worker_running_sum = get_experiment_results('./decay-0.2-batch-0.2/noise-running-sum-decay-0.2-batch-0.2.log', lambda x: extract_from_worker(x, worker))
        worker_ema = get_experiment_results('./decay-0.2-batch-0.2/noise-ema-0.2-batch-0.2.log', lambda x: extract_from_worker(x, worker))
        workers.append((worker_running_sum, worker_ema))

    ax = plt.gca() 
   
    for worker, (worker_logs_running_sum, _) in enumerate(workers):
        plot(ax, worker_logs_running_sum, "Running Sum", worker)

    # With accuracies on train axis
    # plt.gca().invert_xaxis()


    plt.show()



def main():
    paper_figure()

if __name__ == "__main__":
    main()
