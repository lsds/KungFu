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

def extract_val_acc_replicated(s):
    pattern = re.compile(r"\[Global\sStep\s(?P<step>\d+)\]\sAccuracy\s@\s1\s=\s(?P<top1val>\d+\.\d+)\sAccuracy\s@\s5\s=\s(?P<top5val>\d+\.\d+).*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    step    = int(match.group("step"))
    top1val = float(match.group("top1val"))
    top5val = float(match.group("top5val"))

    return ("no-parts", "no-peer", "no-date", step, top1val, top5val)

def get_experiment_results_replicated(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results

def correlate_checkpoint_with_physical_time_for_replicated(at_iteration, log_file):
    def extract_checkpoint_moments(s):
        pattern = re.compile(r"\[\s*(?P<time>\d+\.\d+)\].*\s" +
                             str(at_iteration) + r"\simages/sec:\s.*", re.VERBOSE)
        match = pattern.match(s)
        if match is None:
            return None
        time  = float(match.group("time"))
        print("Iteration: " + str(at_iteration) + ", Time: " + str(time))
        return time


    time = get_experiment_results_replicated(log_file, extract_checkpoint_moments)
    if len(time) == 0:
        return []
    return np.average(time)

def get_replicated_results(log_file_prefix_train, log_file_prefix_validation):
    workers_data = []
    data = get_experiment_results_replicated(log_file_prefix_validation, extract_val_acc_replicated)
    for i in range(len(data)):
        d = data[i]
        time = correlate_checkpoint_with_physical_time_for_replicated(d[3], log_file_prefix_train)
        data[i] = (-2, -2, time, d[3], d[4], d[5])
    data  = data[:-1] # Remove the last iteration
    workers_data.append(data)
    return workers_data