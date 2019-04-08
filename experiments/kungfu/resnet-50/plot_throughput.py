import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator

import re


# Set font
font = {'family' : 'normal',
        'size'   : 19}

matplotlib.rc('font', **font)

# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def extract_avg_throughput(s):
    pattern = re.compile(r".*total\simages/sec:\s(?P<throughput>\d+\.\d+)", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    throughput = float(match.group("throughput"))

    return throughput

def get_experiment_results(log_file, match_function):
    results = []
    with open(log_file, "r") as in_file:
        for line in in_file:
            if (line_regex.search(line)):
                match = match_function(line)
                if match is not None:
                    results.append(match)
    return results


def plotThroughputResNet50Benchmark(avg_throughputs, throughput_ideal):
    strategy = np.array(['Ako(' + str(i) + ')' for i in range(1, 44)])

    parallel_throughput = np.average(np.array([464.95, 464.79, 464.97, 464.90]))

    fig, ax = plt.subplots()

    ax.bar(np.array(['Parallel SGD']), parallel_throughput, align='center', alpha=0.4, color='green')
    ax.bar(np.array(['Parallel SGD']), 138.97 * 4, align='center', alpha=0.4, color='green')


    print(np.max(avg_throughputs))
    print(parallel_throughput)

    ax.bar(strategy, avg_throughputs, align='center', alpha=0.4, color='blue')
    ax.bar(strategy, map(lambda x : x * 4, throughput_ideal), align='center', alpha=0.4)

    plt.ylabel('Images/sec')
    plt.xlabel('Gradient Synchronization Strategy')
    plt.title('ResNet-50 Benchmark using KungFu Parallel SGD and Ako')
    # Use 1 GPU for training. Use CPU <-> GPU communication for synchronization
    # 161 trainable variables in ResNet-50
    # 200 epochs
    # --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=kungfu
    # Next step: print all size partitions of the trainable variables of resnet and compute min size difference between partitions


    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 40))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

    plt.show()


files = np.array(['./logs_resnet_experiments/resnet50-ako-' + str(i) + '.log' for i in range(1, 44)])
results = [get_experiment_results(file, extract_avg_throughput) for file in files]

files_ideal = np.array(['./logs_resnet_experiments_one_process/resnet50-ako-' + str(i) + '.log' for i in range(1, 44)])
results_ideal = [get_experiment_results(file, extract_avg_throughput) for file in files_ideal]


four_peers_throughput = [np.average(res) for res in results]
one_peer_throughput   = [res[0] for res in results_ideal]

plotThroughputResNet50Benchmark(four_peers_throughput, one_peer_throughput)
