"""
BUG: Result cannot be reproduce
"""
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from net import MyNet
from models.resnet import ResNet
from federated import FederatedServer


DATASET = 'cifar-10'
# transformations
TRANSFORM_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066062, ), (0.30810776, ))
])

TRANSFORM_CIFAR10_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_CIFAR10_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

SERVER_SETTINGS = {
    'warm_up': False,
    'setting':{
        'batch_size': 128
    }
}

CLIENT_SETTINGS = {
    'epoch': 3,
    'batch_size': 512
}

if __name__ == '__main__':
    # some settings
    mp.set_start_method('spawn')

    if DATASET == 'mnist':
        net = MyNet
        net_kwargs = None
        dataset = torchvision.datasets.MNIST
        transforms_train = TRANSFORM_MNIST
        transforms_test = TRANSFORM_MNIST
    elif DATASET == 'cifar-10':
        net = ResNet
        net_kwargs = {
            'depth': 32,
            'num_classes': 10
        }
        dataset = torchvision.datasets.CIFAR10
        transforms_train = TRANSFORM_CIFAR10_TRAIN
        transforms_test = TRANSFORM_CIFAR10_TEST
    else:
        raise ValueError('No such dataset. Only have mnist and cifa-10.')

    trainset = dataset('../public_set', train=True, transform=transforms_train, download=True)
    testset = dataset('../public_set', train=False, transform=transforms_test)
    DEVICE_LIST = ['cuda:' + str(i) for i in range(4)]

    # run federated
    fl = FederatedServer(net, trainset, testset, net_kwargs=net_kwargs, devices=DEVICE_LIST, split_method='iid', clients_num=3)
    fl.run(rounds=2)