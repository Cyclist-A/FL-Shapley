"""
BUG: Result cannot be reprocduce
"""
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from models.cnn import CNN as net
from lib.federated import FederatedServer

import torch
import torch.nn as nn
import torch.utils.data as utils
from sklearn.metrics import accuracy_score
# training settings

DEVICE_LIST = ['cuda:' + str(i) for i in range(4)]
ROUNDS = 30

WARM_SETTINGS = {
    'warm_up': False,
    'setting':{
        'batch_size': 128
    }
}

NET_KWARGS = {
    'neurons': 128
}

CLIENT_SETTINGS = {
    'mode': 'thres',
    'epoch': 5,
    'thres': 0.8,
    'batch_size': 16,
    'lr': 1e-3,
    'enable_scheduler': True,
    'eval_each_iter': True,
    'loss_func': nn.CrossEntropyLoss,
    'optimizer': optim.Adam,
    'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_settings':{
        'mode': 'min',
        'factor': 0.13,
        'patience': 3,
        'verbose': True,
        'min_lr': 1e-8,
        'cooldown': 0
    }
}

SERVER_SETTINGS = {
    'split_method': 'iid',
    'clients_num': 3,
    'client_settings': CLIENT_SETTINGS,
    'warm_setting': WARM_SETTINGS,
    'net_kwargs': NET_KWARGS,
    'cal_sv': False,
    'cal_loo': True,
    'devices': DEVICE_LIST
}

TRANSFORM_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066062, ), (0.30810776, ))
])

trainset = torchvision.datasets.MNIST('../public_set', train=True, transform=TRANSFORM_MNIST, download=True)
testset = torchvision.datasets.MNIST('../public_set', train=False, transform=TRANSFORM_MNIST, download=False)

if __name__ == '__main__':
    # some settings
    mp.set_start_method('spawn')
    DEVICE_LIST = ['cuda:' + str(i) for i in range(1, 4)]

    # run federated
    fl = FederatedServer(net, trainset, testset, **SERVER_SETTINGS)
    fl.run(rounds=1)
    print(fl.result)
