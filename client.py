import sys

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
            channel_in: a Queue connected to server, used to send weights to server
            channel_out: a Queue connected to server, used to recevice weights from the server
            dataset: clients dataset
            device: training device
    """
    def __init__(self, net, channel_in, channel_out, dataset, device='cpu'):
        self.net = net
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.dataset = dataset
        self.device = torch.device(device)

        # settings
        self.settings = {
            'epoch': 3, 
            'lr': 0.01,
            'batch_size': 128,
            'loss_func': nn.CrossEntropyLoss,
            'optimizer': optim.Adam,
        }

    def run(self, settings=None):
        """
        Run the client for local training

        ARGS:
            settings: custom settings for local training
        RETURN:
            None
        """
        # customize settings
        if settings:
            for k in settings:
                self.settings[k] = settings[k]

        while True:
            # wait commands from server
            params = self.channel_out.get()

            # finished training
            if type(params) is int:
                break

            # clone and load params
            self.net.load_state_dict(params)
            
            # train local model
            self._train()

            # update to server
            message = {}
            for k in self.net.state_dict():
                message[k] = self.net.state_dict()[k].clone().detach().cpu()
            self.channel_in.put(message)
        
        print(f'Client {params} exited.')
        sys.stdout.flush()

    def run_round(self, params, settings=None):
        """
        Run clients for a round, use for for-loop function

        ARGS:
            params: the state_dict from server
            settings: settings for local training process
        RETURN:
            d_weight(dict): 
        """
        # customize settings
        if settings:
            for k in settings:
                self.settings[k] = settings[k]

        # load params
        self.net.load_state_dict(params)
        
        # train local model
        self._train()

        # calcualte delta weights (no need)
        # d_weights = self._cal_d_weights(params)

        return self.net.state_dict()

    def _train(self):
        """
        Train the model several epochs locally
        
        AGRS:
            None
        RETURN:
            None
        """
        
        # initialize before training
        self.net.to(self.device)
        loader = utils.DataLoader(self.dataset, batch_size=self.settings['batch_size'] , shuffle=True)
        criterion = self.settings['loss_func']()
        optimizer = self.settings['optimizer'](self.net.parameters(), lr=self.settings['lr'])

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
            print(f'Epoch[{epoch + 1}/{self.settings["epoch"]}] Loss: {epoch_loss/i} | Train accu: {train_accu}')
            sys.stdout.flush()

    def _evaluate(self):
        """
        Evaluate current net in the client
        
        ARGS:
            None
        RETURN:
            accuracy_score(float): the 
        """
        self.net.eval()
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
