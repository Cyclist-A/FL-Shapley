import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, neurons=128):
        super().__init__()
        self.conv_out = 245  # gain by error message

        self.conv1 = nn.Conv2d(1, 5, 5)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(5, 5, 5)

        self.bn0 = nn.BatchNorm1d(self.conv_out)
        self.drop_fc0 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(self.conv_out, neurons)
        self.bn1 = nn.BatchNorm1d(neurons)
        self.non_liner = nn.LeakyReLU()

        self.drop_fc1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(neurons, 10)

        # initialize layers
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)

        x = torch.flatten(x, start_dim=1)
        
        x = self.bn0(x)
        x = self.drop_fc0(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.non_liner(x)

        x = self.drop_fc1(x)
        x = self.fc2(x)

        return x
