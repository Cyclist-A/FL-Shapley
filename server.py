import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

import numpy as np
from sklearn.metrics import accuracy_score

class Server:
    """
        Server instance in federated learning
        
        ARGS:
            net: a pyTorch neural network that used by clients and server
            channels: a list of Queues connected to clients
            testset: test the preformance of aggregation weights
            trainset: server's trainset, used as warm up (optional)
            sampler: trainset sampler, used as warm up (optional)
            device: the device name used to warm up and evaluate net
    """
    def __init__(self, net, channels, testset, trainset=None, sampler=None, device='cuda:0'):
        self.device = torch.device(device)
        self.net = net.to(device)
        self.channels = channels
        self.testset = testset
        self.trainset = trainset
        self.sampler = sampler        

    def __del__(self):
        # stop all clients by send a signal -1
        for c in self.channels:
            c.put(-1)

        print('Kill all client, stop training')

        # TODO save the model
        

    def run(self, round=10, c=1, warm_up=False, setting=None, random_state=7):
        """
        Run the server to train model

        ARGS:
            round: total round to run 
            c: proportion of chosen clients in each round
            warm_up: whether to train model before start federated learning
            random_state: random seed
        RETURN:
            None
        """
        # fixed seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # warm up before start training
        if warm_up:
            self._warm_up(setting)

        # start training
        for r in round:
            # choose clients to run
            clients = self._choose_clients(c)
            
            # get the response from chosen clients
            d_params = self._params_from_client(clients)

            # calcualte shapley value TODO

            # aggregate the parameters
            self._step(d_params)

            # evaluate
            self._evaluate()
        
        print('Finished training the model')
        print(f'Test Accu:{self._evaluate()}')

    def _warm_up(self, setting):
        """
        Train server before starting client processes

        ARGS:
            setting: customed setting to warm up
        RETURN:
            None
        """
        default_setting = {
            'epoch': 5, 
            'lr': 0.01,
            'batch_size': 16,
            'loss_func': nn.CrossEntropyLoss,
            'optimizer': optim.Adam,
        }

        # grab settings from
        if setting:
            for k in setting:
                default_setting[k] = setting[k]
        
        # initialize before training
        loader = utils.DataLoader(self.trainset, batch_size=default_setting['batch_size'] ,sampler=self.sampler, shuffle=True)
        criterion = default_setting['loss']()
        optimizer = default_setting['optimizer'](self.net.parameters(), lr=default_setting['lr'])

        print('Start heating server...')

        for epoch in range(default_setting['epoch']):
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
            test_accu = self._evaluate()
            print(f'Epoch[{epoch}/{default_setting["epoch"]}] Test accu:{test_accu}')

    def _choose_clients(self, c):
        """
        Randomly choose some clients to update weight
        
        ARGS:
            c: the proportion of chosen clients, should lie in (0, 1]
        RETURN:
            chosen_clients(list): a list of queues, which are connected to chosen clients
        """
        # select a proportion of client
        if c == 1:
            return self.channels
        elif c > 1 or c <= 0:
            raise ValueError(f'The proportion of chosen clients should lie in (0, 1]. Now is {c}')

        num = max(c * len(self.channels), 1)
        
        chosen_clients = np.random.shuffle(self.channels)[:num]
        return chosen_clients

    def _params_from_client(self, clients):
        """
        Fetch local training resutls from clients

        ARGS:
            clients: the list of chosen clients
        RETURN:
            d_params(list): delta parameters from all chosen clients
        """
        # send params to client by queue
        print('Send parameters to chosen clients...')
        for c in clients:
            c.put(self.net.state_dict())

        # fetch params from client 
        d_params = []
        for c in clients:
            d_params.append(c.get())

        return d_params

    def _step(self, d_params):
        """
        Aggregate delta parameters from different clients.
        Use aggregation resutls to update the model

        ARGS:
            d_params: all parameters from clients
        RETURN:
            None
        """
        # aggregate
        keys = d_params[0].keys()
        aggr_params = {}

        for k in keys:
            aggr_params[k] = d_params[0][k]
            for i in range(len(d_params), 1):
                aggr_params[k] += d_params[i][k]

            aggr_params[k] /= len(d_params)

        # update net's params
        params = self.net.load_state_dict(aggr_params)
        

    def _evaluate(self):
        """
        Evaluate current net in the server
        """
        
        loader = utils.DataLoader(self.testset, batch_size=1000, shuffle=False)
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
