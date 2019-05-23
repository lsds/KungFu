import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re

import random
import datetime

from collections import defaultdict
import math

# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def extract_val_acc(s):
    pattern = re.compile(r"Accuracy\s@\s1\s=\s(?P<top1val>\d+\.\d+)\sAccuracy\s@\s5\s=\s(?P<top5val>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    top1val = float(match.group("top1val"))
    top5val = float(match.group("top5val"))

    return (top1val, top5val)

def extract_val_acc_replicated(s):
    pattern = re.compile(r"\[Global\sStep\s\d+\]\sAccuracy\s@\s1\s=\s(?P<top1val>\d+\.\d+)\sAccuracy\s@\s5\s=\s(?P<top5val>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    top1val = float(match.group("top1val"))
    top5val = float(match.group("top5val"))

    return (top1val, top5val)


def extract_checkpoint_step(s):
    pattern = re.compile(r".*Checkpoint\sat\sstep\s(?P<step>\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    step = int(match.group("step"))

    return step

def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def plot_experiment(steps, data, color, marker, label):
    top1accs, top5accs = zip(*data[::]) 

    plt.plot(steps, top1accs, c=color, marker=marker, markersize=4, ls='-', label=label + ' (Top 1)', fillstyle='none')
    plt.plot(steps, top5accs, c=color, marker=marker, markersize=4, ls='--', label=label + ' (Top 5)', fillstyle='none')

    #plt.errorbar(iters, trainaccs, c=color, yerr=np.array(list(zip(min_accs, max_accs))).T,  marker=marker, fillstyle='none')


def plot_training_accuracy(steps, data_plain, color='r', marker='x', label="No label"):
    # data: tuple (iteration, trainacc)
    plot_experiment(steps, data_plain, color=color, marker=marker, label=label)

def plot_log_files(steps, log_files):
    number_of_colors = 50
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for index, (name, log_file) in enumerate(log_files):
        if name == 'Replicated':
            data = get_experiment_results(log_file, extract_val_acc_replicated)
            data = data[::4] + [data[-1]]
        else:
            data = get_experiment_results(log_file, extract_val_acc)

        plot_training_accuracy(steps, data, color=random.choice(color), marker=random.choice(markers), label=name)

    plt.ylabel('Validation Accuracy')
    plt.xlabel('Iterations')
    plt.title('ResNet-32 Validation Accuracy Over Logical Steps')


    plt.legend(loc='best')

    plt.show()

def main():
    names = [('Ako ' + str(parts) + ' parts', './kungfu-logs-validation/resnet-32-ako-' + str(parts) + '-validation.out') for parts in [15]] # range(1, 33)]
    log_files = [('Parallel SGD', './kungfu-logs-validation/resnet-32-parallel-validation.out'), 
                ('Replicated', './kungfu-logs-validation/resnet-32-replicated-validation-correct.out')] + \
                names[::]
    
    steps = get_experiment_results('./kungfu-logs-validation/resnet-32-b-32-g-1-parallel.out', extract_checkpoint_step)
    # One last time checkpoint
    steps[-1] = steps[-1] // 4

    plot_log_files(steps, log_files)


if __name__ == "__main__":
    main()
