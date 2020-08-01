#!/usr/bin/env python3
'''
# Install PyTorch first and then run the following command
$ pip3 install --no-index -U . # Install KungFu TensorFlow first
$ rm setup.py
$ ln -s setup_pytorch.py setup.py
$ pip3 install --no-index -U . # Install KungFu for PyTorch

$ python3 ./examples/torch_mnist_example.py --epochs 0 --download # Download dataset
$ kungfu-run -np 4 python3 ./examples/torch_mnist_example.py --batch-size 1000 --epochs 3
'''

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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir',
                        type=str,
                        default='data',
                        help='data dir')
    parser.add_argument('--download',
                        action='store_true',
                        default=False,
                        help='download data set')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    return parser.parse_args()


def main():
    # Training settings
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        args.data_dir, train=False, transform=transform),
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **config)

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
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
