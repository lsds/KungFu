import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re

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

def extract_throughput(s):
    pattern = re.compile(r"(?P<experiment>.*)\-(?P<mean>\d+\.\d+)\s*\+\-(?P<std>\d+\.\d+)\s*.*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    experiment = match.group("experiment")
    mean = float(match.group("mean"))
    std = float(match.group("std"))

    return (experiment, mean, std)

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
    parts = match.group("parts")
    stale = match.group("stale")
    kickin = match.group("kickin")

    return "(" + parts + ", " + stale + ", " + kickin  + ")"

def plot_mlp(mlp):
    N = len(mlp)
    mlp  = sorted(mlp, key=lambda tup: tup[1])
    mlpMeans = [exp[1] for exp in mlp]
    mlpStd = [exp[2] for exp in mlp]
    ind = np.arange(N)    # the x locations for the groups

    bars = plt.bar(ind, mlpMeans, 0.8,yerr=mlpStd)
    akoIndex = [i for i in range(len(mlp)) if mlp[i][0] == 'mlp.plain']
    bars[akoIndex[0]].set_color('green')

    plt.ylabel('Throughput (imgs/sec)')
    plt.xlabel('Experiment ID')
    plt.title('Single-Layer Perceptron MNIST Training with Ako and Synchronous SGD Synchronization Strategies (23 trainable variables)')
    # xTicks = ['MLP ' + str(i) for i in range(N)]
    xTicks = [pretty_experiment_name(exp[0]) for exp in mlp]
    plt.xticks(ind, xTicks)
    plt.yticks(np.arange(0, 20, 1))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    legend_dict = {'Ako with configuration (#partitions, staleness, kick-in iteration)' : 'blue', 'Synchronous SGD' : 'green'}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    plt.legend(handles=patchList, loc='upper left')

    plt.show()

def plot_experiment_times():
    results = get_experiment_results('graphdata_experiment_time.txt', extract_experiment_time)
    timeRegex = re.compile(r'.*mlp.*')
    times = filter(lambda x : timeRegex.match(x[0]), results)
    times = [time for time in times]
    N = len(times)
    times  = sorted(times, key=lambda tup: tup[1])
   
    experimentTimes = [t[1] for t in times]
    ind = np.arange(N)    # the x locations for the groups

    bars = plt.bar(ind, experimentTimes, 0.8)
    akoIndex = [i for i in range(len(times)) if times[i][0] == 'mlp.plain']
    bars[akoIndex[0]].set_color('green')

    plt.ylabel('Experiment time (sec)')
    plt.xlabel('Experiment ID')
    plt.title('Multi-Layer Perceptron MNIST Training Times with Ako and Synchronous SGD Synchronization Strategies')
    # xTicks = ['MLP ' + str(i) for i in range(N)]
    xTicks = [pretty_experiment_name(exp[0]) for exp in times]
    plt.xticks(ind, xTicks)
    plt.yticks(np.arange(0, 20000, 1000))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    legend_dict = {'Ako with configuration (#partitions, staleness, kick-in iteration)' : 'blue', 'Synchronous SGD' : 'green'}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    plt.legend(handles=patchList, loc='upper left')

    plt.show()

def plot_ako_vs_plain(log_file):
    results = get_experiment_results(log_file, extract_throughput)
    # mlpRegex = re.compile(r'mlp\..*')
    # mlp = filter(lambda x : mlpRegex.match(x[0]), results)
    # mlp = [x for x in mlp]
    
    #plot_mlp(mlp)
    plot_experiment_times()

def main():
    plot_ako_vs_plain("graphdata_throughput.txt")

if __name__ == "__main__":
    main()