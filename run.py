import torch.multiprocessing as mp
import torchvision

from net import MyNet
from federated import Federated

# transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.13066062, ), (0.30810776, ))
])

# some settings
NET = MyNet
TRAINSET = torchvision.datasets.MNIST('../public_set', train=True, transform=transform, download=True)
TESTSET = torchvision.datasets.MNIST('../public_set', train=False, transform=transform)
DEVICE_LIST=['cuda:1']

SERVER_SETTINGS = {
    'warm_up': False,
    'setting':{
        'batch_size': 128
    }
}

# CLIENT_SETTINGS = {}

if __name__ == '__main__':
    mp.set_start_method('spawn')
    fl = Federated(NET, TRAINSET, TESTSET, DEVICE_LIST, split_method='imba-size')
    fl.run(server_settings=SERVER_SETTINGS)
    # fl.run_for_loop()
