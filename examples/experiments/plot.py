import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re

# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def extract_throughput(s):
    pattern = re.compile(r"(?P<experiment>.*)\-(?P<mean>\d+\.\d+)\s*\+\-(?P<std>\d+\.\d+)\s*.*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    experiment = match.group("experiment")
    mean = float(match.group("mean"))
    std = float(match.group("std"))

    return (experiment, mean, std)

def get_experiment_results(log_file):
    results = []
    # Open input file in 'read' mode
    with open(log_file, "r") as in_file:
        # Loop over each log line
        for line in in_file:
            # If log line matches our regex, print to console, and output file
            if (line_regex.search(line)):
                match = extract_throughput(line)
                if match is not None:
                    results.append(match)
    return results

def pretty_experiment_name(experiment_name):
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

def plot_slp(slp):
    N = len(slp)
    slp  = sorted(slp, key=lambda tup: tup[1])
    slpMeans = [exp[1] for exp in slp]
    slpStd = [exp[2] for exp in slp]
    ind = np.arange(N)    # the x locations for the groups

    bars = plt.bar(ind, slpMeans, 0.8,yerr=slpStd)
    akoIndex = [i for i in range(len(slp)) if slp[i][0] == 'slp.plain']
    bars[akoIndex[0]].set_color('green')

    plt.ylabel('Throughput (imgs/sec)')
    plt.xlabel('Experiment ID')
    plt.title('Multi-Layer Perceptron MNIST Training with Ako and Synchronous SGD Synchronization Strategies (23 trainable variables)')
    # xTicks = ['SLP ' + str(i) for i in range(N)]
    xTicks = [pretty_experiment_name(exp[0]) for exp in slp]
    plt.xticks(ind, xTicks)
    plt.yticks(np.arange(0, 50000, 2500))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    legend_dict = {'Ako with configuration (#partitions, staleness, kick-in iteration)' : 'blue', 'Synchronous SGD' : 'green'}
    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    plt.legend(handles=patchList, loc='upper left')

    plt.show()

def plot_ako_vs_plain(log_file):
    results = get_experiment_results(log_file)
    slpRegex = re.compile(r'slp\..*')
    slp = filter(lambda x : slpRegex.match(x[0]), results)
    slp = [x for x in slp]
    mlpRegex = re.compile(r'mlp\..*')
    mlp = filter(lambda x : mlpRegex.match(x[0]), results)
    mlp = [x for x in mlp]
    
    plot_mlp(mlp)
    #plot_slp(slp)

def main():
    plot_ako_vs_plain("graphdata_throughput.txt")

if __name__ == "__main__":
    main()