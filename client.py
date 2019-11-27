import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

from sklearn.metrics import accuracy_score

class Client:
    """
        Client instance in federated learning
        
        ARGS:
            net: a pyTorch neural network that used by clients and server
            channel: a Queue connected to server
            dataset: clients dataset
            device: training device
            sampler: sampler for dataset loader
            settings: custom settings for local training
    """
    def __init__(self, net, channel, dataset, device='cpu', sampler=None, settings=None):
        self.net = net
        self.channel = channel
        self.dataset = dataset
        self.device = torch.device(device)
        self.sampler = sampler

        # settings
        self.settings = {
            'epoch': 5, 
            'lr': 0.01,
            'batch_size': 16,
            'loss_func': nn.CrossEntropyLoss,
            'optimizer': optim.Adam,
        }

        if settings:
            for k in settings:
                self.settings[k] = settings[k]

    def run(self):
        """
        Run the client for local training

        ARGS:
            None
        RETURN:
            None
        """
        # wait commands from server:
        while True:
            params = self.channel.get()

            # finished training
            if type(params) is int and params == -1:
                break

            # load params
            self.net.load_state_dict(params)
            
            # train local model
            self._train()

            # calcualte delta weights
            d_weights = self._cal_d_weights(params)

            # update to server
            self.channel.put(d_weights)

    def _train(self):
        """
        Train the model several epochs locally
        
        AGRS:
            None
        RETURN:
            None
        """
        
        # initialize before training
        loader = utils.DataLoader(self.dataset, batch_size=self.settings['batch_size'] ,sampler=self.sampler, shuffle=True)
        criterion = self.settings['loss']()
        optimizer = self.settings['optimizer'](self.net.parameters(), lr=self.settings['lr'])

        print('Start heating server...')

        for epoch in range(self.settings['epoch']):
            self.net.train()
            epoch_loss = 0

            for i, data in enumerate(loader, 1):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)

                # updata
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            # evaluate
            train_accu = self._evaluate()
            print(f'Epoch[{epoch}/{self.settings["epoch"]}] Train accu:{train_accu}')

    def _cal_d_weights(self, params):
        """
        Calculate the different between server's parameters and client;s parameters after training

        ARGS:
            params: params from server
        RETURN:
            d_params(dict): difference of the parameters
        """
        new_param = self.net.state_dict()
        d_params = {}
        
        for k in params:
            d_params[k] = new_param[k] - params[k]
        
        return d_params

    def _evaluate(self):
        """
        Evaluate current net in the client
        
        ARGS:
            None
        RETURN:
            accuracy_score(float): the 
        """
        
        loader = utils.DataLoader(self.dataset, batch_size=1000, shuffle=False)
        predicted = []
        truth = []

        for data in loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.net(inputs)
            _, pred = torch.max(outputs.data, 1)
            
            for p, q in zip(pred, labels):
                predicted.append(p.item())
                truth.append(q.item())
        
        return accuracy_score(truth, predicted)
