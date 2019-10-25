#!/usr/bin/env python3

import argparse

import numpy as np

import gym
from diff_trainer import DiffTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='RL example.')
    parser.add_argument('--agent', type=str, default='mlp', help='agent name')
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='sgd | adam | rmsp')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='batch size')
    parser.add_argument('--episodes',
                        type=int,
                        default=50000,
                        help='max number of episodes')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='',
                        help='path to checkpoint file')
    parser.add_argument('--init',
                        type=bool,
                        default=False,
                        help='generate init checkpoint')
    return parser.parse_args()


def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float)


def init_agent(env, agent, opt):
    init = env.reset()
    image = prepro(init)
    if agent == 'cnn':
        from cnn_agent import Agent
    else:
        # use mlp as default
        from mlp_agent import Agent

    a = Agent(image.shape, env.action_space, opt)
    return a


def main():
    args = parse_args()
    env = gym.make("Pong-v0")
    agent = init_agent(env, args.agent, args.optimizer)
    if args.init:
        print('saving checkpoint %s' % args.checkpoint)
        agent.save(args.checkpoint)
        return
    elif args.checkpoint:
        print('loading checkpoint %s' % args.checkpoint)
        agent.load(args.checkpoint)

    trainer = DiffTrainer(args.batch_size, prepro)
    trainer.train(agent, env, args.episodes)


main()
