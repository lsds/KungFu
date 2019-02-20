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

# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def extract_experiment_time(s):
    pattern = re.compile(r"(?P<experiment>.*):train\stook\s(?P<time>\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    experiment = match.group("experiment")
    time = float(match.group("time"))

    return (experiment, time)

def extract_validation_acc_ako(s):
    pattern = re.compile(r"ako.(?P<parts>\d+)parts.*peer-(?P<peer>\d+).stdout.*\s-\svalidation\saccuracy\s\(epoch\s(?P<epoch>\d+)\):\s(?P<valacc>\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    parts = int(match.group("parts"))
    peer  = int(match.group("peer"))
    epoch = int(match.group("epoch"))
    valacc = float(match.group("valacc"))
    
    matchTime = re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', s)
    date = datetime.datetime.strptime(matchTime.group(), '%Y-%m-%d %H:%M:%S')

    return (parts, peer, date, epoch, valacc)

def extract_time_to_acc_plain(s):
    pattern = re.compile(r"plain.*peer-(?P<peer>\d+).stdout.*target\s(?P<target>\d+\.\d+).*\(time\s(?P<time>\d+\.\d+)\)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    peer  = int(match.group("peer"))
    target = float(match.group("target"))
    time = float(match.group("time"))
    
    return (peer, target, time)

def extract_time_to_acc_ako(s):
    pattern = re.compile(r"ako.(?P<parts>\d+)parts.*peer-(?P<peer>\d+).stdout.*target\s(?P<target>\d+\.\d+).*\(time\s(?P<time>\d+\.\d+)\)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    parts = int(match.group("parts"))
    peer  = int(match.group("peer"))
    target = float(match.group("target"))
    time = float(match.group("time"))
    
    return (parts, peer, target, time)



def extract_validation_acc_plain(s):
    pattern = re.compile(r"plain.*peer-(?P<peer>\d+).stdout.*\s-\svalidation\saccuracy\s\(epoch\s(?P<epoch>\d+)\):\s(?P<valacc>\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    peer  = int(match.group("peer"))
    epoch = int(match.group("epoch"))
    valacc = float(match.group("valacc"))
    
    matchTime = re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', s)
    date = datetime.datetime.strptime(matchTime.group(), '%Y-%m-%d %H:%M:%S')

    return (peer, date, epoch, valacc)


def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def pretty_experiment_name(experiment_name):
    print(experiment_name)
    if 'plain' in experiment_name:
        return 'Synch SGD'
    pattern = re.compile(r".*\.(?P<parts>\d+).*\.(?P<stale>\d+).*\.kickin(?P<kickin>\d+)", re.VERBOSE)
    match = pattern.match(experiment_name)
    if match is None:
        return None
    parts  = match.group("parts")
    stale  = match.group("stale")
    kickin = match.group("kickin")

    return "(" + parts + ", " + stale + ", " + kickin  + ")"


def plot_ako(data_ako):
    ako_dict = defaultdict(lambda: defaultdict(lambda:  defaultdict(int)))
    for parts, peer, date, epoch, valacc in data_ako:
        ako_dict[parts][epoch][peer] = valacc

    res = defaultdict(list)

    for parts in range(1, 34):
        exp_result = ako_dict[parts]
        for epoch in range(10):
            avg_val_acc = 0
            min_val_acc = math.inf
            max_val_acc = 0
            for peer in range(4):
                avg_val_acc += exp_result[epoch][peer]
                min_val_acc = min(min_val_acc, exp_result[epoch][peer])
                max_val_acc = max(max_val_acc, exp_result[epoch][peer])
            avg_val_acc /= 4
            res[parts].append((epoch, avg_val_acc, min_val_acc, max_val_acc))

    import random

    number_of_colors = 7
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    epochsToValAccs = defaultdict(list)

    for index, parts in enumerate([1, 5, 10, 15, 20, 25, 33]):
        tups = res[parts]
        epochs, val_acc, min_accs, max_accs = [[ i for i, j, k, l in tups ], [ j for i, j, k, l in tups ],  [ j - min_acc for i, j, min_acc, l in tups ],
                           [ max_acc - j for i, j, k, max_acc in tups ]]
        for i, epoch in enumerate(epochs):
            epochsToValAccs[epoch].append(val_acc[i])
        plt.plot(epochs, val_acc, c=color[index], marker=markers[index], ls='-', label="#partitions " + str(parts), fillstyle='none')
        plt.errorbar(epochs, val_acc, c=color[index], yerr=np.array(list(zip(min_accs, max_accs))).T,  marker=markers[index], fillstyle='none')

    # medianValAccs = []
    # for epoch, valAccs in epochsToValAccs.items():
    #     medianValAccs.append(np.median(valAccs))
    
    # plt.plot(epochs, medianValAccs, c='black', marker='P', ls='-', label="Ako Median Validation Accuracy", fillstyle='none')

    # for each epoch, for each partition

def plot_plain(data_plain):
    ako_dict = defaultdict(lambda:  defaultdict(int))
    for peer, date, epoch, valacc in data_plain:
        ako_dict[epoch][peer] = valacc
  
    tups = []

    for epoch in range(10):
        avg_val_acc = 0
        min_val_acc = math.inf
        max_val_acc = 0
        for peer in range(4):
            avg_val_acc += ako_dict[epoch][peer]
            min_val_acc = min(min_val_acc, ako_dict[epoch][peer])
            max_val_acc = max(max_val_acc, ako_dict[epoch][peer])
        avg_val_acc /= 4
        tups.append((epoch, avg_val_acc, min_val_acc, max_val_acc))


    import random

    number_of_colors = 7
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    epochs, val_acc, min_accs, max_accs = [[ i for i, j, k, l in tups ], [ j for i, j, k, l in tups ],  [ j - min_acc for i, j, min_acc, l in tups ],
                        [ max_acc - j for i, j, k, max_acc in tups ]] 
    plt.plot(epochs, val_acc, c='r', marker=markers[10], ls='-', label="Synch SGD", fillstyle='none')
    plt.errorbar(epochs, val_acc, c='r', yerr=np.array(list(zip(min_accs, max_accs))).T,  marker=markers[10], fillstyle='none')


def plot_validation_accuracy(data_ako, data_plain):
    # data_ako   : tuple (parts, peer, date, epoch, valacc)
    # data_plain : tuple (peer, date, epoch, valacc)

    plot_ako(data_ako)
    plot_plain(data_plain)

    plt.ylabel('Validation accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('LeNet (MNIST) Average Peer Validation Accuracy with Ako and Synchronous SGD Synchronization Strategies (33 trainable variables)')
    

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(0.9, 1.0, 0.01))

    plt.legend(loc='bottom right')

    plt.show()

def plot_time_to_accuracy(data_ako, data_plain):
    ako_dict = defaultdict(lambda:  defaultdict(int))

    globalTarget = 0 
    for parts, peer, target, time in data_ako:
        globalTarget = target
        ako_dict[parts][peer] = time

    tups = []

    for parts in range(1, 34):
        exp_result = ako_dict[parts]
        times = []
        for peer in range(4):
            times.append(exp_result[peer])
        tups.append((parts, times))

    num_partitions, times = [[ i for i, j in tups ], [ j for i, j in tups ]]

    ind = range(0, 34)
    ts =  [np.mean([51.95504546165466, 51.7059862613678, 51.50437092781067])] + [np.mean(ts) for ts in times]
    std = [np.std([51.95504546165466, 51.7059862613678, 51.50437092781067])] + [np.std(ts) for ts in times]
    print(ts)
    print(len(ts))
    print(len(std))
    bars = plt.bar(ind, ts, 0.8, yerr=std)
    bars[0].set_color('green')

    xTicks = ["S-SGD"] + ["#" + str(parts) for parts, times in tups]
    plt.xticks(range(0, 34), xTicks)
    
    plt.yticks(np.arange(0, 300, 50))

    plt.ylabel("Time to target accuracy " + str(globalTarget) + "%")
    plt.xlabel('Experiment Configuartion')
    plt.title('LeNet (MNIST) Time to Accuracy with Ako and Synchronous SGD Synchronization Strategies')

    # legend_dict = {'Ako with configuration (#partitions, staleness, kick-in iteration)' : 'blue', 'Synchronous SGD' : 'green'}
    # patchList = []
    # for key in legend_dict:
    #     data_key = mpatches.Patch(color=legend_dict[key], label=key)
    #     patchList.append(data_key)

    # plt.legend(handles=patchList, loc='upper left')

    plt.show()


def plot_ako_vs_plain_validation_accuracy(log_file):
    data_ako = get_experiment_results(log_file, extract_validation_acc_ako)
    # tuple (parts, peer, date, epoch, valacc)
    data_plain = get_experiment_results(log_file, extract_validation_acc_plain)
    # tuple (peer, date, epoch, valacc)
    plot_validation_accuracy(data_ako, data_plain)
    

def plot_ako_vs_plain_time_to_accuracy(log_file):
    data_ako = get_experiment_results(log_file, extract_time_to_acc_ako)
    # tuple (parts, peer, target, time)
    data_plain = get_experiment_results(log_file, extract_time_to_acc_plain)
    # tuple (peer, target, time)
    plot_time_to_accuracy(data_ako, data_plain)



def main():
    plot_ako_vs_plain_validation_accuracy("val_acc.txt")
    plot_ako_vs_plain_time_to_accuracy("time_to_acc.txt")
    

if __name__ == "__main__":
    main()
