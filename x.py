import os
import subprocess
import sys

os.environ['RUNNER'] = os.environ['USER']
os.environ['SRC_DIR'] = os.path.dirname(os.path.realpath(__file__)) # pwd
os.environ['EXPERIMENT_SCRIPT'] = './examples/LeNet.py'

subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "init-remote"])
subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "prepare"])

def build_args(strategy, batch_size, n_epochs, partitions=None, staleness=None, kickin=None):
    if strategy ==  'ako':
        return "--kungfu-strategy %s --batch-size %s --n-epochs %s --ako-partitions %d --staleness %d --kickin-time %d" % \
                (strategy, batch_size, n_epochs, partitions, staleness, kickin)
    else:
        return  "--batch-size %s --n-epochs %s" % (batch_size, n_epochs)
def set_args(args):
    os.environ['EXPERIMENT_ARGS'] = args

def build_log_name(strategy, batch_size, n_epochs, partitions=None, staleness=None, kickin=None):
    if strategy ==  'ako':
        return "%s.%dparts.%dstale.%dkickin.%dbatch.%depochs" % (strategy, partitions, staleness, kickin, batch_size, n_epochs)
    else:
        return strategy
def set_log_name(name):
    os.environ['PRETTY_EXPERIMENT_NAME'] = name

def run_experiments(n_epochs):
    for strategy in config_grid['strategy']: 
       for batch_size in config_grid['batch']:
         for parts in config_grid['parts']:
             for stale in config_grid['stale']: 
                 for kickin in config_grid['kickin']:
                     if strategy == 'ako':
                         print('The args are: ' + build_args(strategy, batch_size, n_epochs, partitions=parts, staleness=stale, kickin=kickin))
                         set_args(build_args(strategy, batch_size, n_epochs, partitions=parts, staleness=stale, kickin=kickin))
                         set_log_name(build_log_name(strategy, batch_size, n_epochs, partitions=parts, staleness=stale, kickin=kickin))
                         subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"])
                     else:
                         print('The args are ' + build_args(strategy, batch_size, n_epochs))
                         set_args(build_args(strategy, batch_size, n_epochs))
                         set_log_name(build_log_name(strategy, batch_size, n_epochs))
                         subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"]) 
                         return

os.environ['EXPERIMENT_SCRIPT'] = './examples/LeNet.py'
# LeNet5, 33 variables
config_grid = {'strategy': ['ako', 'plain'], 
               'batch'   : [8], # [1, 2, 4, 8, 16, 32],
               'parts'   : [1, 2, 3],  # [i for i in range(1, 12)],
               'stale'   : [0], # unused
               'kickin'  : [0]}

run_experiments(n_epochs=1)