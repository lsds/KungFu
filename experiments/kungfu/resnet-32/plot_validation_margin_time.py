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

def extract_val_acc(s):
    pattern = re.compile(r"\[Global\sStep\s(?P<step>\d+)\]\sAccuracy\s@\s1\s=\s(?P<top1val>\d+\.\d+)\sAccuracy\s@\s5\s=\s(?P<top5val>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    step    = int(match.group("step"))
    top1val = float(match.group("top1val"))
    top5val = float(match.group("top5val"))

    return ("no-parts", "no-peer", "no-date", step, top1val, top5val)

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


def plot_ako(current_partition, current_partition_index, data_ako, partitions, peers, epochs):
    ako_dict = defaultdict(lambda: defaultdict(lambda:  defaultdict(int)))

    data_ako = [d for sublist in data_ako for d in sublist]

    for parts, peer, phys_time, epoch, top1valacc, top5valacc in data_ako:
        ako_dict[parts][epoch][peer] = (top1valacc, phys_time)

    res = defaultdict(list)

    for parts in partitions:
        exp_result = ako_dict[parts]
        for epoch in epochs:
            avg_val_acc = 0
            min_val_acc = float('inf')
            max_val_acc = 0
            for peer in peers:
                avg_val_acc += exp_result[epoch][peer][0]
                min_val_acc = min(min_val_acc, exp_result[epoch][peer][0])
                max_val_acc = max(max_val_acc, exp_result[epoch][peer][0])
            avg_val_acc /= len(peers)
            res[parts].append((epoch, avg_val_acc, min_val_acc, max_val_acc, exp_result[epoch][peer][1]))

    number_of_colors = 100
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    epochsToValAccs = defaultdict(list)

    tups = res[current_partition]
    epochs, val_acc, min_accs, max_accs, times = [[ i for i, j, k, l, m in tups ], 
                                                  [ j for i, j, k, l, m in tups ],  
                                                  [ j - min_acc for i, j, min_acc, l, m in tups ],
                                                  [ max_acc - j for i, j, k, max_acc, m in tups ],
                                                  [ m for i, j, k, l, m in tups ]]
    for i, epoch in enumerate(epochs):
        epochsToValAccs[epoch].append(val_acc[i])

    if current_partition == -1:
        label = 'Parallel SGD'
    else:
        label = "#partitions " + str(current_partition)

    print(times)

    plt.plot(times, val_acc, c=color[current_partition_index], marker=markers[current_partition_index], ls='-', label=label, fillstyle='none')
    plt.errorbar(times, val_acc, c=color[current_partition_index], yerr=np.array(list(zip(min_accs, max_accs))).T,  marker=markers[current_partition_index], fillstyle='none')

    # medianValAccs = []
    # for epoch, valAccs in epochsToValAccs.items():
    #     medianValAccs.append(np.median(valAccs))
    
    # plt.plot(epochs, medianValAccs, c='black', marker='P', ls='-', label="Ako Median Validation Accuracy", fillstyle='none')

    # for each epoch, for each partition

def plot_validation_accuracy(current_partition, current_partition_index, data_ako, partitions, peers, epochs):
    # data_ako   : tuple (parts, peer, date, epoch, valacc)
    # data_plain : tuple (peer, date, epoch, valacc)

    plot_ako(current_partition, current_partition_index, data_ako, partitions, peers, epochs)
    # plot_plain(data_plain)

    plt.ylabel('Validation accuracy (%)')
    plt.xlabel('Physical Time (seconds)')
    plt.title('ResNet-32 Average Top 1 Peer Validation Accuracy over Physical Time with Ako and Parallel SGD Synchronization Strategies')
    

    plt.xticks(np.arange(1, 1600, 100))
    plt.yticks(np.arange(0.0, 1.1, 0.1))


def correlate_checkpoint_with_physical_time(from_worker, at_iteration, log_file):
    def extract_checkpoint_moments(s):
        pattern = re.compile(r"\[.*/0" + str(from_worker) + r"/.*stdout\]\s\[\s*(?P<time>\d+\.\d+)\].*" +
                             str(at_iteration) + r"\simages/sec:\s.*", re.VERBOSE)
        match = pattern.match(s)
        if match is None:
            return None
        time  = float(match.group("time"))
        return time


    time = get_experiment_results(log_file, extract_checkpoint_moments)
    if len(time) == 0:
        return []
    return np.average(time)

def get_ako_results(parts, log_file_prefix_train, log_file_prefix_validation):
    workers_data = []
    for worker in range(0, 4):
        data = get_experiment_results(log_file_prefix_validation + str(worker), extract_val_acc)
        for i in range(len(data)):
            d = data[i]
            time = correlate_checkpoint_with_physical_time(worker, d[3], log_file_prefix_train)
            data[i] = (parts, worker, time, d[3], d[4], d[5])
        data  = data[:-1] # Remove the last iteration
        workers_data.append(data)
    return workers_data

def plot_by_iterations():
    # -1 means KungFu Parallel SGD
    partitions = [4, 5, 7, 10, 11, 12, 14, -1] #  range(4, 15)

    ako_files = []
    for parts in partitions[:-1]:
        ako_files.append(("exhaustive/training/resnet-32-b-32-g-1-ako-" + str(parts) + "-partitions.out",
                          "exhaustive/validation/validation-ako-" + str(parts) + "-worker-"))

    # Parallel
    ako_files.append(("exhaustive/training/resnet-32-b-32-g-1-parallel.out",
                      "exhaustive/validation/validation-parallel-worker-"))

    for current_partition_index, current_partition in enumerate(partitions):
        log_train = ako_files[current_partition_index][0]
        log_valid = ako_files[current_partition_index][1]
        data_ako = get_ako_results(current_partition, log_train, log_valid)    
        
        # tuple (parts, peer, date, epoch, valacc)
        parts    = list(set([d[0] for peer_data in data_ako for d in peer_data]))
        peers    = list(set([d[1] for peer_data in data_ako for d in peer_data]))
        epochs   = [d[3] for d in data_ako[0]]

        
        plot_validation_accuracy(current_partition, current_partition_index, data_ako, parts, peers, epochs)
   

def plot_ako_vs_plain_validation_accuracy():
    plot_by_iterations()
    plt.legend(loc='upper left')

    plt.show()

    
def main():
    plot_ako_vs_plain_validation_accuracy()

if __name__ == "__main__":
    main()
