import argparse

import tensorflow as tf
from tensorflow.python.util import deprecation

import rlzoo

deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-l', type=int, default=1)
    p.add_argument('-a', type=int, default=1)
    p.add_argument('-s', type=int, default=1)
    return p.parse_args()


def example_save_request(agent):
    x = tf.Variable(1, dtype=tf.float32)
    print(x)
    agent.save(x)
    print('saved')
    agent.barrier()
    print('barrier done')
    y = agent.request(rlzoo.Role.Leaner, 0, x)
    print(y)


def range1(n):
    return range(1, n + 1)


def train(agent):
    n_steps = 1
    for i in range1(n_steps):
        print('step %d' % (i))
        x = tf.Variable([10], dtype=tf.float32)
        y = agent.all_reduce(x)
        print(x)
        print(y)


def run_leaner(agent):
    train(agent)


def run_actor(agent):
    pass


def run_server(agent):
    pass


def main():
    args = parse_args()
    agent = rlzoo.Agent(n_leaners=args.l, n_actors=args.a, n_servers=args.s)
    print('%s : %d/%d' % (agent.role(), agent.rank(), agent.size()))
    # example_save_request(agent)

    if agent.role() == rlzoo.Role.Leaner:
        run_leaner(agent)
    elif agent.role() == rlzoo.Role.Actor:
        run_actor(agent)
    elif agent.role() == rlzoo.Role.Server:
        run_server(agent)
    else:
        raise RuntimeError('invalid role')

    agent.barrier()


print('BEGIN')
main()
print('END')
