import os
import subprocess
import sys

os.environ['RUNNER'] = os.environ['USER']
os.environ['SRC_DIR'] = os.path.dirname(os.path.realpath(__file__)) # pwd
os.environ['EXPERIMENT_SCRIPT'] = './examples/LeNet.py'

subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "init-remote"])
subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "prepare"])

def build_args(strategy, partitions=None, staleness=None, kickin=None):
    if strategy ==  'ako':
        return "--kungfu-strategy %s --ako-partitions %d --staleness %d --kickin-time %d" % (strategy, partitions, staleness, kickin)
    else:
        return ""
def set_args(args):
    os.environ['EXPERIMENT_ARGS'] = args

def build_log_name(strategy, partitions=None, staleness=None, kickin=None):
    if strategy ==  'ako':
        return "%s.%dparts.%dstale.%dkickin" % (strategy, partitions, staleness, kickin)
    else:
        return strategy
def set_log_name(name):
    os.environ['PRETTY_EXPERIMENT_NAME'] = name

# Logistic regression experiments, 3 variables
# config_grid = {'strategy': ['ako', 'plain'], 
#                'parts'   : [1, 2, 3],
#                'stale'   : [i  for i in range(0, 900, 25)], 
#                'kickin'  : [i  for i in range(0, 900, 25)]}

# LeNet5, 33 variables
config_grid = {'strategy': ['ako'], 
               'parts'   : [10, 20, 30],
               'stale'   : [0], # unused
               'kickin'  : [0]}


def run_logistic_regression():
    for strategy in config_grid['strategy']: 
        for parts in config_grid['parts']:
            for stale in config_grid['stale']: 
                for kickin in config_grid['kickin']:
                    if strategy == 'ako':
                        set_args(build_args(strategy=strategy, partitions=parts, staleness=stale, kickin=kickin))
                        set_log_name(build_log_name(strategy=strategy, partitions=parts, staleness=stale, kickin=kickin))
                        subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"])
                    else:
                        set_args(build_args(strategy))
                        set_log_name(build_log_name(strategy))
                        subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"]) 
                        return

run_logistic_regression()