import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator


def variateNumberOfMachines():
    machines = np.array([1, 2, 3, 4], dtype=int)
    throughputs = [215.2, 357.8, 538.8, 653.7]

    # One machine one GPU
    throughputIdeal = 56.2

    fig, ax = plt.subplots()

    ax.bar(machines, throughputs, align='center', alpha=0.4, color='blue')
    ax.bar(machines, map(lambda x : throughputIdeal * x * 4, machines), align='center', alpha=0.4)
    plt.ylabel('Images/sec')
    plt.xlabel('Number of 4-GPU machines')
    plt.title('Horovod Training with Real Data on Tesla M60 GPU Machines')

    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 40))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

def variateBatchSize():
    batches = np.array([8, 16, 32, 64], dtype=int)
    throughputs = [181.1, 330.6, 653.7, 832.8]

    # One machine one GPU
    throughputIdeal8  = 48.8
    throughputIdeal16 = 53.6
    throughputIdeal32 = 56.2
    throughputIdeal64 = 57.0
    ideal = [throughputIdeal8, throughputIdeal16, throughputIdeal32, throughputIdeal64]


    fig, ax = plt.subplots()

    xi = [i for i in range(0, len(batches))]
    ax.bar(xi, throughputs, align='center', alpha=0.4, color='blue')
    ax.bar(xi, map(lambda x : x * 4 * 4, ideal), align='center', alpha=0.4)
    plt.ylabel('Images/sec')
    plt.xlabel('Batch Size')
    plt.title('Horovod Training with Real Data on 4 Tesla M60 GPU Machines')

    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 50))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xticks(xi, batches)

    plt.show()


variateNumberOfMachines()
#variateBatchSize()
