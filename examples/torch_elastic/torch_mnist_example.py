#!/usr/bin/env python3
"""
# Install PyTorch first and then run the following command
$ git checkout -f setup.py
$ pip3 install --no-index -U . # Install KungFu TensorFlow first
$ rm setup.py
$ ln -s setup_pytorch.py setup.py
$ pip3 install --no-index -U . # Install KungFu for PyTorch

# run example
$ ./examples/torch_elastic/run.sh
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import kungfu.python as kfpy


class SLP(nn.Module):
    def __init__(self, input_size, logits):
        super(SLP, self).__init__()
        self._input_size = input_size
        self.fc = nn.Linear(input_size, logits)

    def forward(self, x):
        x = x.view(-1, self._input_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_step(model, optimizer, data, label, device):
    data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, label)
    loss.backward()
    optimizer.step()
    return loss


def sync_model(model):
    import kungfu.torch as kf
    kf.broadcast_parameters(model.state_dict())


def train(args, model, device, optimizer, step_based_schedule):
    model.train()
    train_loader = create_datasets(args)

    it = enumerate(train_loader)

    def get_next(it):
        while True:
            try:
                return next(it), it
            except StopIteration:
                it = enumerate(train_loader)

    max_step = args.epochs * 60000 // args.batch_size

    need_sync = True
    step = 0
    while step < max_step:
        if need_sync:
            step = kfpy.all_reduce_int_max(step)
            sync_model(model)
            need_sync = False

        (_batch_idx, (data, label)), it = get_next(it)
        loss = train_step(model, optimizer, data, label, device)
        print('step %d loss: %f' % (step, loss.item()))

        if step in step_based_schedule:
            new_size = step_based_schedule[step]
            need_sync, detached = kfpy.resize(new_size)
            if detached:
                break

        step += 1


class MNIST(datasets.MNIST):
    prefix = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    resources = [
        (prefix + "train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        (prefix + "train-labels-idx1-ubyte.gz",
         "d53e105ee54ea40749a09fcbcd1e9432"),
        (prefix + "t10k-images-idx3-ubyte.gz",
         "9fb629c4189551a2d022fa330f9573f3"),
        (prefix + "t10k-labels-idx1-ubyte.gz",
         "ec29112dd5afa0611ce80d1b7f02629c"),
    ]


def create_datasets(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    config = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_loader = torch.utils.data.DataLoader(
        MNIST(
            args.data_dir,
            train=True,
            download=args.download,
            transform=transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **config,
    )
    return train_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64, metavar='N')
    p.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    p.add_argument('--epochs', type=int, default=10, metavar='N')
    p.add_argument('--lr', type=float, default=0.01, metavar='LR')
    p.add_argument('--momentum', type=float, default=0.5, metavar='M')
    p.add_argument('--no-cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=1, metavar='S')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--download', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    model = SLP(28 * 28, 10).to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum)

    # BEGIN kungfu
    import kungfu.torch as kf
    optimizer = kf.optimizers.SynchronousSGDOptimizer(
        optimizer, named_parameters=model.named_parameters())
    # END kungfu

    step_based_schedule = {
        100: 2,
        200: 3,
        300: 4,
        400: 2,
        500: 3,
        600: 1,
    }
    train(args, model, device, optimizer, step_based_schedule)


if __name__ == '__main__':
    main()
