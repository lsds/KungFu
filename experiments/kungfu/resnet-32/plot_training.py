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

def extract_training_acc(s):
    pattern = re.compile(r"(?P<iter>\d+)\simages/sec:\s(\d+\.\d+)\s\+/-\s(\d+\.\d+)\s\(jitter\s=\s(\d+\.\d+)\)\s(\d+\.\d+)\s(?P<top1trainacc>\d+\.\d+)\s(\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    iteration = int(match.group("iter"))
    trainacc = float(match.group("top1trainacc"))

    return (iteration, trainacc)

def extract_training_acc_multipeer(s):
    pattern = re.compile(r".*/00-of-04.*\s(?P<iter>\d+)\simages/sec:\s(\d+\.\d+)\s\+/-\s(\d+\.\d+)\s\(jitter\s=\s(\d+\.\d+)\)\s(\d+\.\d+)\s(?P<top1trainacc>\d+\.\d+)\s(\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    iteration = int(match.group("iter"))
    trainacc = float(match.group("top1trainacc"))

    return (iteration, trainacc)

def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def plot_experiment(ax, indices_pair, data, color, marker, label):
    iters, trainaccs = zip(*data[::25]) 
    ax[indices_pair[0], indices_pair[1]].plot(iters, trainaccs, c=color, marker=marker, markersize=2, ls='-', label=label, fillstyle='none')
    #plt.errorbar(iters, trainaccs, c=color, yerr=np.array(list(zip(min_accs, max_accs))).T,  marker=marker, fillstyle='none')


def plot_training_accuracy(ax, indices_pair, data_plain, color='r', marker='x', label="No label"):
    # data: tuple (iteration, trainacc)
    plot_experiment(ax, indices_pair, data_plain, color=color, marker=marker, label=label)

def plot_log_files(ax, indices_pair, log_files):
    number_of_colors = 50
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for index, (name, log_file) in enumerate(log_files):
        if name == 'Replicated':
            data = get_experiment_results(log_file, extract_training_acc)
        else:
            data = get_experiment_results(log_file, extract_training_acc_multipeer)
        plot_training_accuracy(ax, indices_pair, data, color=random.choice(color), marker=random.choice(markers), label=name)

    ax[indices_pair[0], indices_pair[1]].legend(loc='best')

    # plt.show()

def main():
    # Default three experiments
    # names = [('Ako ' + str(parts) + ' parts', './kungfu-logs/baseline-comparison-logs/resnet32-ako-' + str(parts) + '.log') for parts in [15]] # range(1, 33)]
    # plot_log_files([('Parallel SGD', './kungfu-logs/baseline-comparison-logs/resnet32-parallel-sgd.log'), ('Replicated', './kungfu-logs/baseline-comparison-logs/resnet32-replicated.log')] + names[::])

    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    indices = [(i, j) for i in range(2) for j in range(3)]

    for i, ps in enumerate([4, 5, 10, 15, 20, 25]):
        names = [('Ako ' + str(parts) + ' parts', './kungfu-logs-training/ako-partition-range-logs/resnet32-ako-' + str(parts) + '-partitions-span-experiment.log') 
                 for parts in [ps]] # range(1, 33)]
        plot_log_files(ax, indices[i], 
        [('Parallel SGD', './kungfu-logs-training/baseline-comparison-logs/resnet32-parallel-sgd.log'), 
        ('Replicated', './kungfu-logs-training/baseline-comparison-logs/resnet32-replicated.log')] + 
        names[::])

    ax[1, 0].set_ylabel('Training accuracy')
    ax[0, 0].set_ylabel('Training accuracy')
    ax[1, 1].set_xlabel('Iterations')
    plt.suptitle('ResNet-32 Convergence')


    plt.xticks(np.arange(0, 65000, 5000))
    plt.yticks(np.arange(0.0, 1.05, 0.1))

    ax[1, 2].set_xticks(np.arange(0, 60000, 10000))
    ax[0, 0].set_yticks(np.arange(0.0, 1.05, 0.1))

    plt.show()

if __name__ == "__main__":
    main()
