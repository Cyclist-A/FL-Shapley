import torch.multiprocessing as mp
import torchvision

from net import MyNet
from federated import FederatedServer

# transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.13066062, ), (0.30810776, ))
])

# some settings
NET = MyNet
TRAINSET = torchvision.datasets.MNIST('../public_set', train=True, transform=transform, download=True)
TESTSET = torchvision.datasets.MNIST('../public_set', train=False, transform=transform)
DEVICE_LIST=['cuda:' + str(i) for i in range(1, 4)]

SERVER_SETTINGS = {
    'warm_up': False,
    'setting':{
        'batch_size': 128
    }
}

# CLIENT_SETTINGS = {}

if __name__ == '__main__':
    mp.set_start_method('spawn')
    fl = FederatedServer(NET, TRAINSET, TESTSET, devices=DEVICE_LIST, split_method='iid', C=3)
    fl.run()
