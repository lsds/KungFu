import re

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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


def extract_from_worker(s, from_worker):
    pattern = re.compile(r"\[.*/0" + str(from_worker) + r"/.*\].*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return s


def print_none_match(pattern, l):
    print(pattern.pattern)
    raise Exception(l)


def get_loss(s):
    pattern = re.compile(r".*\)\t(?P<loss>\d+\.\d+)\t.*", re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return float(match.group("loss"))


def get_noise(s):
    pattern = re.compile(r".*iteration\[(?P<noise>[\-\+]?\d+\.\d+)\]\n",
                         re.VERBOSE)
    match = pattern.match(s)
    if match is None:
        return None
    return float(match.group("noise"))


def plot(lines):
    losses = [get_loss(l) for l in lines]
    losses = filter(None, losses)

    noises = [get_noise(l) for l in lines]
    noises = filter(None, noises)

    pairs = zip(losses, noises)
    pairs.sort(key=lambda x: x[0])

    print(pairs)

    losses, noises = zip(*pairs)

    plt.ylim([-200000, 200000])
    plt.title('ResNet-32 gradient noise scale')
    plt.ylabel('Gradient Noise')
    plt.xlabel('Training Loss')

    plt.plot(losses, noises)

    plt.show()


def main():
    num_workers = 1
    workers = []
    for worker in range(num_workers):
        worker = get_experiment_results(
            './correctnoise-tensorboard.log',
            lambda x: extract_from_worker(x, worker))
        workers.append(worker)

    for worker_logs in workers:
        plot(worker_logs)


if __name__ == "__main__":
    main()
