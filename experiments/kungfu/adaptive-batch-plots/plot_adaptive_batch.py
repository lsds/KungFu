import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import os
import re

import datetime


from collections import defaultdict
import math
import random

import itertools

from matplotlib.pyplot import figure



# Set size of figure
figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')


font = {'family' : 'normal',
        'weight' : '300',
        'size'   : 17}

matplotlib.rc('font', **font)


# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results



def extract_val_acc(s):
    pattern = re.compile(r"\[Global\sStep\s(?P<step>\d+)\]\sAccuracy\s@\s1\s=\s(?P<top1val>\d+\.\d+)\sAccuracy\s@\s5\s=\s(?P<top5val>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    step    = int(match.group("step"))
    top1val = float(match.group("top1val"))
    top5val = float(match.group("top5val"))

    return top1val


def plot(times, accs):
    plt.plot(times, accs, color='b', label="Adaptive", marker='x')

def get_training_times_and_accs(s):
    from_worker = 0
    pattern = re.compile(r"\[.*/0" + str(from_worker) + r"/.*stdout\]\s\[\s*(?P<time>\d+\.\d+)\].*\simages/sec:\s.*", re.VERBOSE)

    match = pattern.match(s)
    if match is None:
        return None
    time  = float(match.group("time"))
    return time


def get_training_times_and_accs_simple(s):
    from_worker = 0

    if "Checkpoint at step" in s:
        s =  s.strip('\n')
        split = list(s.split(" "))
        index = split.index("step")
        return "Step " + split[index + 1]

    pattern = re.compile(r"\[.*/0" + str(from_worker) + r"/.*stdout\]\s\[\s*(?P<time>\d+\.\d+)\].*\simages/sec:\s.*", re.VERBOSE)

    match = pattern.match(s)
    if match is None:
        return None
    time  = float(match.group("time"))
    return time

def get_step_val_acc(s):
    if "Global Step" in s:
        s = s.split(" ")
        return (int(s[2][:-1]), float(s[7]))    
    return None

def plot_static(train_file, validation_file, color, label, marker):
    train_times = get_experiment_results(train_file, lambda x : get_training_times_and_accs_simple(x))
    step_to_second = dict()
    prev = train_times[0]
    for time in train_times:
        if type(time) is str and "Step" in time:
            step_to_second[int(time.split(" ")[1])] = prev
        else:
            prev = time
    
    step_val_acc = get_experiment_results(validation_file, lambda x : get_step_val_acc(x))
    times_accs = []
    for s, acc in step_val_acc:
        times_accs.append((step_to_second[s], acc))

    times, accs = zip(*times_accs)


    if "4096" in label:
        times = times[:-30]
        accs = accs[:-30]

    times = times
    accs = accs

    print("Max of " + label + ": " + str(np.max(accs)))
    plt.plot(times, accs, color=color, label=label, marker=marker)


def build_train_val_files(run_id):
    return ("training-adaptive-reproduced-no-warm-up/training-parallel-" + str(run_id), "validation-adaptive-reproduced-no-warm-up/validation-parallel-worker-0-validation")


def main():
    vals = []
    _, val_file = build_train_val_files(-1)
    vals   = get_experiment_results(val_file, lambda x: extract_val_acc(x))
    session_times = []
    for run_id in range(1, 26):
        train_file, _ = build_train_val_files(run_id)
        train_times = get_experiment_results(train_file, lambda x : get_training_times_and_accs(x))
        session_times.append(train_times[-1])

    times = list(itertools.accumulate(session_times))

    plot(times, vals)


    #plt.axhline(y=vals[8], xmin=0, xmax=100, color='red')


    #plot_static("training-static/training-parallel", "validation-static/validation-parallel-worker-0-validation")
    l1 = "B = 4096"
    plot_static("training-static-1024/training-parallel", "validation-static-1024/validation-parallel-worker-0-validation", 'sienna', l1, '^')
    l2 = "B = 256"
    plot_static("training-static-64/training-parallel", "validation-static-64/validation-parallel-worker-0-validation", 'slategrey', l2, 'p')

    plt.ylabel("Validation Accuracy (%)")
    plt.xlabel("Time (seconds)")



    plt.xticks(np.arange(0, 500, 50))
    #plt.xticks(np.arange(0, 2500, 200))
    plt.yticks(np.arange(0.1, 1.1, 0.1))



    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:d}'.format(int(x * 100)) for x in vals])
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)


    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
