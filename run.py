"""
BUG: Result cannot be reprocduce
"""
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from net import MyNet
from models.resnet import ResNet
from federated import FederatedServer

import argparse

parser = argparse.ArgumentParser(description='Federated Learning Framework')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dataset', type=str, default='cifar-10', metavar='N',
                    help='set up different datasets')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='enable warm up')
parser.add_argument('--split-method', type=str, default='iid', metavar='N',
                    help='split the dataset with different assumptions')
parser.add_argument('--num-workers', type=int, default=3, metavar='N',
                    help='the number of workers in FL(default: 3)')
parser.add_argument('--num-rounds', type=int, default=1, metavar='N',
                    help='the rounds of aggregation(default: 1)')



# transformations
TRAINSFORM_MINST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066062, ), (0.30810776, ))
])

TRAINSFORM_CIFA10_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRAINSFORM_CIFA10_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CLIENT_SETTINGS = {}

def main(args):
    # some settings
    mp.set_start_method('spawn')

    if args.dataset == 'mnist':
        net = MyNet
        net_kwargs = None
        dataset = torchvision.datasets.MNIST
        transforms_train = TRAINSFORM_MINST
        transforms_test = TRAINSFORM_MINST
    elif args.dataset == 'cifar-10':
        net = ResNet
        net_kwargs = {
            'depth': 32,
            'num_classes': args.num_classes
        }
        dataset = torchvision.datasets.CIFAR10
        transforms_train = TRAINSFORM_CIFA10_TRAIN
        transforms_test = TRAINSFORM_CIFA10_TEST
    else:
        raise ValueError('No such dataset. Only have mnist and cifa-10.')

    trainset = dataset('../public_set', train=True, transform=transforms_train, download=True)
    testset = dataset('../public_set', train=False, transform=transforms_test)
    DEVICE_LIST = ['cuda:' + str(i) for i in range(1, 4)]

    # run federated
    fl = FederatedServer(net, trainset, testset, net_kwargs=net_kwargs, devices=DEVICE_LIST, split_method=args.split_method, clients_num=args.num_workers)
    fl.run(rounds=args.num_rounds)

if __name__ == "__main__":
    args = parser.parse_args()
    SERVER_SETTINGS = {
        'warm_up': args.warm_up,
        'setting': {
            'batch_size': args.batch_size
        }
    }
    main(args)
