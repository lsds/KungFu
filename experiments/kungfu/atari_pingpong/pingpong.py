#!/usr/bin/env python3

import argparse
import os
import time

import gym
import numpy as np

from diff_trainer import DiffTrainer
from mlp_agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(description='RL example.')
    parser.add_argument(
        '--batch-size', type=int, default=10, help='batch size')
    parser.add_argument(
        '--episodes', type=int, default=10000, help='max number of episodes')
    return parser.parse_args()


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def init_agent(env):
    init = env.reset()
    image = prepro(init)
    a = Agent(image.shape, env.action_space)
    a.save('pingpong-init.npz')
    checkpoing = 'pingpong.latest.npz'
    if os.path.isfile(checkpoing):
        a.load(checkpoing)
    return a


def main():
    args = parse_args()
    env = gym.make("Pong-v0")
    agent = init_agent(env)
    trainer = DiffTrainer(args.batch_size, prepro)
    trainer.train(agent, env, args.episodes)


main()
