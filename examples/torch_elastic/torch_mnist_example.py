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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(model, optimizer, data, target, device)
        # TODO: resize cluster
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def create_datasets(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    config = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir,
                       train=True,
                       download=args.download,
                       transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        **config)

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
    p.add_argument('--log-interval', type=int, default=10, metavar='N')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--download', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    train_loader = create_datasets(args)

    model = SLP(28 * 28, 10).to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum)

    # BEGIN kungfu
    import kungfu.torch as kf
    optimizer = kf.optimizers.SynchronousSGDOptimizer(
        optimizer, named_parameters=model.named_parameters())
    kf.broadcast_parameters(model.state_dict())
    # END kungfu

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)


if __name__ == '__main__':
    main()
