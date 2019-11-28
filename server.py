import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils

import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

class Server:
    """
        Server instance in federated learning
        
        ARGS:
            net: a pyTorch neural network that used by clients and server
            channels: a list of Queues connected to clients
            testset: test the preformance of aggregation weights
            trainset: server's trainset, used as warm up (optional)
            device: the device name used to warm up and evaluate net
    """
    def __init__(self, net, channels, testset, trainset=None, device='cuda:0'):
        self.device = torch.device(device)
        self.net = net.to(device)
        self.current_params = self.net.state_dict()
        self.testset = testset
        self.trainset = trainset

        # set idx for channels, use as identity
        self.channels = {i: c for i, c in enumerate(channels)}

    def __del__(self):
        # stop all clients by send a signal -1
        for idx in self.channels:
            self.channels[idx].put(-1)

        print('Kill all client, stop training')

        # TODO save the model
        

    def run(self, round=10, c=1, warm_up=False, setting=None, random_state=7):
        """
        Run the server to train model TODO

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
        for r in range(round):
            # choose clients to run
            clients = self._choose_clients(c)
            
            # get the response from chosen clients
            d_params = self._params_from_client(clients)

            # calcualte shapley value
            self._shapley_value_sampling(d_params)

            # aggregate the parameters
            self._step(d_params)

            # evaluate
            test_accu = self._evaluate()
            print(f'Round[{r+1}/{round}] Test Accu: {test_accu}')
        
        print('Finished training the model')
        print(f'Test Accu:{self._evaluate()}')

    def _warm_up(self, settings):
        """
        Train server before starting client processes

        ARGS:
            settings: customed settings to warm up
        RETURN:
            None
        """
        default_setting = {
            'epoch': 5, 
            'lr': 0.01,
            'batch_size': 128,
            'loss_func': nn.CrossEntropyLoss,
            'optimizer': optim.Adam,
        }

        # grab settings from
        if settings:
            for k in settings:
                default_setting[k] = settings[k]
        
        # initialize before training
        loader = utils.DataLoader(self.trainset, batch_size=default_setting['batch_size'] , shuffle=True)
        criterion = default_setting['loss_func']()
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
            print(f'Epoch[{epoch+1}/{default_setting["epoch"]}] Loss: {epoch_loss/i} | Test accu:{test_accu}')
        
        # store the training result
        self.current_params = self.net.state_dict()

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
        
        # randomly choose clients idx
        num = max(c * len(self.channels), 1)
        idx = [i for i in range(len(self.channels))]
        np.random.shuffle(idx)
        idx = idx[:num]
        
        # choose clients by idx
        chosen_clients = {i: self.channels[i] for i in idx}

        return chosen_clients

    def _params_from_client(self, clients):
        """
        Fetch local training resutls from clients

        ARGS:
            clients: the dict of chosen clients
        RETURN:
            d_params(dict): delta parameters from all chosen clients
        """
        # send params to client by queue
        print('Send parameters to chosen clients...')
        for key in clients:
            clients[key].put(self.current_params)

        # fetch params from client 
        d_params = {}
        for key in clients:
            while True:
                if not clients[key].empty():
                    break
                time.sleep(5)
            d_params[key] = clients[key].get()

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
        idx = d_params.keys()

        layers = d_params[idx[0]].keys()
        aggr_params = {}

        for l in layers:
            aggr_params[l] = d_params[idx[0]][l]
            for i in idx[1:]:
                aggr_params[l] += d_params[i][l]

            aggr_params[l] /= len(d_params)

        # update net's params
        self.current_params = aggr_params

    def _evaluate(self, params=None):
        """
        Evaluate current net in the server
        ARGS:
            params: params for self.net

        RETURN:
            accuracy_score: evalutaion result 
        """
        # For empty set in Shapley Value, it should return 0
        if params and len(params) == 0:
            return 0.0

        elif not params:
            params = self.current_params

        loader = utils.DataLoader(self.testset, batch_size=1000, shuffle=False)
        predicted = []
        truth = []

        self.net.load_state_dict(params)
        self.net.eval()

        with torch.no_grad():
            for data in loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # outputs = model(inputs)
                outputs = self.net(inputs)
                _, pred = torch.max(outputs.data, 1)

                for p, q in zip(pred, labels):
                    predicted.append(p.item())
                    truth.append(q.item())

        return accuracy_score(truth, predicted)

    def _aggregate(self, weights):
        """
        :param
            weights:  a list contains workers' weights
        RETURN:
            aggr_params(dict): a aggregated weights
        """
        if not weights:
            return {}

        keys = weights[0].keys()
        aggr_params = {}

        for k in keys:
            aggr_params[k] = weights[0][k]
            for i in range(len(weights), 1):
                aggr_params[k] += weights[i][k]

            aggr_params[k] /= len(weights)

        return aggr_params

    def _shapley_value_sampling(self, d_params, samples=1000):
        """
        Calculate Shapley Values for clients
        
        ARGS:
            d_params:
            samples: sampling times
        RETURN:
            shapley(dict): Client weights' shapely valye
        """
        w_ids = d_params.keys()
        N = len(w_ids)
        result = defaultdict(float)
        for r in range(samples):
            p = np.random.permutation(w_ids)
            for i in range(p):
                y = [d_params[_id] for _id in p[:i+1]]
                y0 = [d_params[_id] for _id in p[:i]]
                u_y, u_y0 = self._evaluate(self._aggregate(y)), self._evaluate(self._aggregate(y0))
                delta = u_y - u_y0
                result[p[i]] += delta
        shapley = result / samples
        return shapley