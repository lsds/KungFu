import os
import subprocess
import sys

# Change paths according to lcoation of this experiment script
os.environ['RUNNER'] = os.environ['USER']
os.environ['SRC_DIR'] = os.path.dirname(os.path.realpath(__file__)) # pwd
os.environ['EXPERIMENT_SCRIPT'] = 'examples/lenet.py'
os.environ['KUNGFU_CONFIG_ENABLE_MONITORING'] = 'true'
os.environ['KUNGFU_CONFIG_MONITORING_PERIOD'] = '10ms'

subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "init-remote"])
subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "prepare"])

def build_args(strategy, batch_size, n_epochs, partitions=None):
    if strategy ==  'ako':
        return "--kungfu-strategy %s --batch-size %s --n-epochs %s --ako-partitions %d" % \
                (strategy, batch_size, n_epochs, partitions)
    else:
        return  "--batch-size %s --n-epochs %s" % (batch_size, n_epochs)
def set_args(args):
    os.environ['EXPERIMENT_ARGS'] = args

def build_log_name(strategy, batch_size, n_epochs, partitions=None):
    if strategy ==  'ako':
        return "%s.%dparts.%dbatch.%depochs" % (strategy, partitions, batch_size, n_epochs)
    else:
        return strategy
def set_log_name(name):
    os.environ['PRETTY_EXPERIMENT_NAME'] = name

def run_experiments(n_epochs):
    for strategy in config_grid['strategy']: 
       for batch_size in config_grid['batch']:
         for parts in config_grid['parts']:
                     if strategy == 'ako':
                         print('The args are: ' + build_args(strategy, batch_size, n_epochs, partitions=parts))
                         set_args(build_args(strategy, batch_size, n_epochs, partitions=parts))
                         set_log_name(build_log_name(strategy, batch_size, n_epochs, partitions=parts))
                         subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"])
                     else:
                         print('The args are ' + build_args(strategy, batch_size, n_epochs))
                         set_args(build_args(strategy, batch_size, n_epochs))
                         set_log_name(build_log_name(strategy, batch_size, n_epochs))
                         subprocess.call(["./KungFu/scripts/azure/relay-machine/run-experiments.sh", "run"]) 
                         return

os.environ['EXPERIMENT_SCRIPT'] = 'examples/lenet.py'
# LeNet5, 33 variables
config_grid = {'strategy': ['ako'], 
               'batch'   : [8], # [1, 2, 4, 8, 16, 32] Pick batch size 8 because it generates more traffic
               'parts'   : [5], # [i for i in range(1, 12)],
               }

run_experiments(n_epochs=1)