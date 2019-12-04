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
C = 3
TRAINSET = torchvision.datasets.MNIST('.', train=True, transform=transform, download=True)
TESTSET = torchvision.datasets.MNIST('.', train=False, transform=transform)
DEVICE_LIST=['cuda:0']

SERVER_SETTINGS = {
    'warm_up': True,
    'setting':{
        'batch_size': 128
    }
}

# CLIENT_SETTINGS = {}

if __name__ == '__main__':
    fl = Federated(NET, C, TRAINSET, TESTSET, DEVICE_LIST)
    fl.run_for_loop()
