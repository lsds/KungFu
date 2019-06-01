import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

import os
import re
import math
import random
import datetime

line_regex = re.compile(r".*$")



def plot_pie(bucket_sizes, ax, current_partition):
    data = [float(x.split()[0]) for x in bucket_sizes]
    ingredients = [x.split()[-1] for x in bucket_sizes]

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d} bytes)".format(pct, absolute)
    
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                    textprops=dict(color="w"))
    ax.legend(wedges, ingredients,
            title="Partitions",
            loc="center left",
            bbox_to_anchor=(1.3, 0, 0.5, 1))
    plt.setp(autotexts, size=6, weight="bold")
    ax.set_title("{0:.0%}".format(current_partition) +  " Partition Budget for ResNet-32")


def extract_partitions(s):
    pattern = re.compile(r".*01/01-of-04.*Partition{budget=(?P<budget>\d+),\scurrent_cost=(?P<partition_cost>\d+)}.*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    budget    = int(match.group("budget"))
    cost = float(match.group("partition_cost"))

    return (budget, cost)

def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results


def get_ako_results(parts, log_file_prefix_train):
    return get_experiment_results(log_file_prefix_train, extract_partitions)



def get_files(partial_exchange_type, partitions):
    ako_files = []
    worker = -1
    for parts in partitions:
        ako_files.append(("training/kungfu-logs-" + partial_exchange_type + "/resnet-32-b-32-g-1-"+ 
                           partial_exchange_type + "-" + 
                           str(parts) + "-fraction.out",
                          "validation/kungfu-logs-" + partial_exchange_type + 
                          "/validation-" + partial_exchange_type + "-fraction-" +
                           str(parts) + "-worker-"))                      
    return ako_files


def plot_by_iterations():
    partitions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ako_files = get_files("partial_exchange", partitions)
    # ako_files = get_files("partial_exchange_accumulation", partitions)
    # ako_files = get_files("partial_exchange_accumulation_avg_peers", partitions)
    # ako_files = get_files("partial_exchange_accumulation_avg_window", partitions)

    fig, ax = plt.subplots(3, 3, sharex="col", sharey="row", subplot_kw=dict(aspect="equal"))

    indices = [(i, j) for i in range(3) for j in range(3)]

    for current_partition_index, current_partition in enumerate(partitions):
        log_train = ako_files[current_partition_index][0]
        data_ako = get_ako_results(current_partition, log_train)    
        data_ako = (data_ako[0][0], [cost for budget, cost in data_ako])
        data_ako = ["{:.1f} bytes Partition {:d}".format(size, index + 1) for index, size in enumerate(data_ako[1])]
        plot_pos = indices[current_partition_index]
        plot_pie(data_ako, ax[plot_pos[0], plot_pos[1]], current_partition)

        # tuple (parts, peer, date, epoch, valacc)
        # parts   = list(set([d[0] for peer_data in data_ako for d in peer_data]))
        # peers   = list(set([d[1] for peer_data in data_ako for d in peer_data]))
        # epochs  = [d[3] for d in data_ako[0]]

        # plot_validation_accuracy(current_partition, current_partition_index, data_ako, parts, peers, epochs)
    fig.suptitle("ResNet-32 Bin Packing Gradient Placement", fontsize=16)
    plt.show()


plot_by_iterations()