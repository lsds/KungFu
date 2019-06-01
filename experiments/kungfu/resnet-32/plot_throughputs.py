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

m = dict()
curr_partitions = 1

def extract_average_throughput(s):
    global curr_partitions
    global m
    pattern = re.compile(r".*total\simages/sec:\s(?P<throughput>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        pattern = re.compile(r".*--ako_partitions=(?P<partitions>\d+).*", re.VERBOSE)
        match = pattern.match(s)
        if match is not None:
           parts =  int(match.group("partitions")) 
           curr_partitions = parts
        return None
    
    thr = float(match.group("throughput"))

    if curr_partitions not in m:
        m[curr_partitions] = [thr]
    else:
        m[curr_partitions].append(thr)

    return thr

def match_throughput(iteration, s):
    global curr_partitions
    global m
    pattern = re.compile(r".*" + re.escape(str(iteration)) + r"\timages/sec:\s(?P<throughput>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    
    thr = float(match.group("throughput"))

    return thr


def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def plot_throughputs(throughputs):
    number_of_colors = 10
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for name, dictionary in throughputs:
        keys, values = dictionary.keys(), map(lambda x: np.average(x), dictionary.values())
        plt.plot(keys, values, c=random.choice(color), marker=random.choice(markers), markersize=4, ls='-', label=name, fillstyle='none')

    plt.hlines(2197, 1, 33, label="TensorFlow Independent")

    plt.yticks(np.arange(0, 2500, 100))
    plt.xticks(np.arange(1, 34, 1))

    plt.ylabel('Training Throughput (images/sec)')
    plt.xlabel('Number of Ako Partitions')
    plt.title('ResNet-32 Training Throughput Average on 4 Workers')

    plt.legend(loc='lower left')

    plt.show()



def plot_partial_exchange(throughputs):
    number_of_colors = 50
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for name, dictionary in throughputs:
        keys, values = dictionary.keys(), map(lambda x: np.average(x), dictionary.values())
        keys = [0.1 * k for k in keys]
        plt.plot(keys, values, c=random.choice(color), marker=random.choice(markers), markersize=4, ls='-', label=name, fillstyle='none')

    plt.hlines(7872, 0.1, 0.9, label="TensorFlow Independent")

# def plot_ako_throughputs():

#     no_op_results = [] 
#     for parts in range(1, 34):
#         throughput_it_1562 = get_experiment_results('./ako-throughput-analysis/tf-operator-noop/all-ako-throughputs-no-op-' + str(parts) + '-partitions.out',
#                                                     match_throughput_it_1562)
#         no_op_results.append(np.average(throughput_it_1562))
#     no_op_results[-2] = 1670
#     no_op_results[-1] = 1665
#     no_op_lst  =  no_op_results

#     no_acc_lst = [1444.9, 1493.5, 1524.4, 1484.5, 1506.6, 1509.0, 1497.1, 1477.6, \
#                  1495.4, 1458.1, 1473.9, 1532.5, 1483.6, 1495.3, 1443.5, 1535.7, \
#                  1429.8, 1530.2, 1475.0, 1428.0, 1458.1, 1450.0, 1501.2,\
#                  1428.8, 1404.3, 1408.9, 1432.3, 1441.7, 1392.7, 1452.9,\
#                  1452.3, 1488.0, 1455.6]
#     summation_lst = [1146.7, 1296.6, 1353.7, 1400.1, 1391.6, 1390.5, 1428.0, 1388.6,
#                     1439.4, 1411.4, 1427.3, 1428.1, 1460.8, 1418.4, 1426.8,
#                     1433.2, 1454.1, 1398.4, 1413.4, 1403.8, 1396.7, 1400.7,
#                     1403.6, 1347.9, 1392.7, 1418.2, 1418.2 , 1398.5, 1401.0,
#                     1374.2, 1399.6, 1397.3, 1410.9]
    
#     averaging_lst = [1038.3,1207.7,1244.9,1295.7,1307.7,1300.8,1339.4,1323.4,1312.6,
# 1338.9,1328.8,1327.3,1344.1,1326.6,1353.6,
# 1369.1,1366.9,1337.5,1340.2,1363.9,1343.7,1342.5,
# 1319.6,1336.5,1323.2,1318.8,1298.8,1332.3,1302.0,1289.9,1289.8,1304.6,1328.7]

#     no_op = dict([(i, no_op_lst[i - 1]) for i in range(1, 34)])
#     no_acc = dict([(i, no_acc_lst[i - 1]) for i in range(1, 34)])
#     summation = dict([(i, summation_lst[i - 1]) for i in range(1, 34)])
#     averaging = dict([(i, averaging_lst[i - 1]) for i in range(1, 34)])

#     names = [('Ako no accumulation', no_acc),
#             ('New TF Operator no-op', no_op),
#             ('Ako with Accumulation (summation)', summation),
#             ('Ako with Accumulation (averaging)', averaging)]
#     plot_throughputs(names)

################################## Partial Exchange ##########################################

def get_partial_exchange_files(partial_exchange_type):
    partitions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ako_files = []
    for parts in partitions:
        ako_files.append("partial-exchange-variants/training/kungfu-logs-" + partial_exchange_type + "/resnet-32-b-32-g-1-"+ 
                            partial_exchange_type + "-" + 
                            str(parts) + "-fraction.out")
    return ako_files

def get_partial_exchange_thrs(partial_exchange_type):
    partial_exchange_files = get_partial_exchange_files(partial_exchange_type)
    partial_exchange_thrs = [] 
    for file in partial_exchange_files:
        throughput_it_1562 = get_experiment_results(file,
                                                    lambda line: match_throughput(1562, line))
        partial_exchange_thrs.append(np.sum(throughput_it_1562))
    return partial_exchange_thrs


def change_name(name):
    m = dict([("partial_exchange", "Gradient Bin Packing Partial Exchange"), 
              ("partial_exchange_accumulation", "Gradient Bin Packing Partial Exchange with History"), 
              ("partial_exchange_accumulation_avg_peers", "Gradient Bin Packing Partial Exchange with History and All-reduce Peer Count Average"),
              ("partial_exchange_accumulation_avg_window", "Gradient Bin Packing Partial Exchange with History and All-reduce Running Average")])
    return m[name]


def plot_replicated_throughput():
    throughput_it_1562_replicated = 4036
    plt.hlines(throughput_it_1562_replicated, 0.1, 0.9, color='brown', label="TensorFlow Replicated")

def plot_partial_exchange_throughputs():
    names = []

    for partial_exchange_type in ["partial_exchange", "partial_exchange_accumulation", 
                              "partial_exchange_accumulation_avg_peers"]:
                              #"partial_exchange_accumulation_avg_window"]:
        lst = get_partial_exchange_thrs(partial_exchange_type)
        lst = dict([(i, lst[i - 1]) for i in range(1, 10)])
        names.append((change_name(partial_exchange_type), lst))
    
    plot_partial_exchange(names)
    plot_replicated_throughput()

    plt.yticks(np.arange(0, 8200, 250))
    plt.xticks(np.arange(0, 1, 0.1))

    plt.ylabel('Training Throughput (images/sec)')
    plt.xlabel('Bucket Budget')
    plt.title('ResNet-32 Aggregated Training Throughput for Gradient Bin Packing Partial Exchange and TensorFlow Strategies')

    plt.legend(loc='lower left')

    plt.show()

plot_partial_exchange_throughputs()

