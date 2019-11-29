import sys
import math
import itertools

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
        
    def run(self, round=3, c=1, warm_up=False, setting=None, random_state=7):
        """
        Run the server to train model

        ARGS:
            round: total round to run 
            c: proportion of chosen clients in each round
            warm_up: whether to train model before start federated learning
            random_state: random seed
        RETURN:
            Noneit
        """
        # fixed seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # warm up before start training
        if warm_up:
            self._warm_up(setting)

        # start training
        print('Federated, Start!')
        sys.stdout.flush()
        
        for r in range(round):
            # choose clients to run
            clients = self._choose_clients(c)
            
            # get the response from chosen clients
            params = self._params_from_client(clients)

            # calcualte shapley value
#             self._shapley_value_sampling(d_params)

            # aggregate the parameters
            self._step(params)

            # evaluate
            test_accu = self._evaluate()
            print(f'Round[{r+1}/{round}] Test Accu: {test_accu}')
            sys.stdout.flush()
        
        print('Finished training the model')
        print(f'Final Test Accu:{self._evaluate()}')
        sys.stdout.flush()
        
        # kill all clients 
        for idx in self.channels:
            self.channels[idx].put(idx)

        print('Kill all clients, stop training')
        sys.stdout.flush()

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
        sys.stdout.flush()

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
            sys.stdout.flush()
        
        # store the training result
        self.current_params = self.net.state_dict()

    def _choose_clients(self, c):
        """
        Randomly choose some clients to update weight
        
        ARGS:
            c: the proportion of chosen clients, should lie in (0, 1]
        RETURN:
            chosen_clients(dict): client_id: queue . Queues are connected to chosen clients
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
            params(dict): delta parameters from all chosen clients
        """
        # send params to client by queue
        print('Send parameters to chosen clients...')
        sys.stdout.flush()
        
        message = {k: self.current_params[k].clone().detach().cpu() for k in self.current_params}
        
        for key in clients:
            clients[key].put(message)

        # fetch params from client 
        params = {}
        for key in clients:
            params[key] = clients[key].get()

        return params

    def _step(self, params):
        """
        Aggregate delta parameters from different clients.
        Use aggregation resutls to update the model

        ARGS:
            params: all parameters from clients
        RETURN:
            None
        """
        # aggregate
        idx = list(params.keys())

        layers = params[idx[0]].keys()
        aggr_params = {}

        for l in layers:
            aggr_params[l] = params[idx[0]][l]
            for i in idx[1:]:
                aggr_params[l] += params[i][l]

            aggr_params[l] /= len(params)

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

        aggr_p = {}

        for k in weights[0].keys():
            aggr_p[k] = weights[0][k]
            for i in range(1, len(weights)):
                aggr_p[k] += weights[i][k]

            aggr_p[k] /= len(weights)
        for k in aggr_p.keys():
            print("after aggregation: ", aggr_p[k])
            break
        return aggr_p

    def _shapley_value_sampling(self, d_params, samples):
        """
        Calculate Shapley Values for clients

        ARGS:
            d_params:
        RETURN:
            result(dict): Client weights' shapely valye
        """
        w_ids = list(d_params.keys())
        N = len(w_ids)
        # samples = math.factorial(N) * 0.1 if N > 10 else math.factorial(N)
        result = defaultdict(float)
        for p in itertools.permutations(w_ids, N):
            print("sampling: ", p)
            sv_pre = 0.0
            for cur in range(len(p)):
                sv_cur = self._evaluate(self._aggregate([d_params[wk_id] for wk_id in p[:cur+1]]))
                print("cur SV: ", sv_cur)
                result[p[cur]] += (sv_cur - sv_pre)
                print("%d worker's sv %.6f" % (p[cur], result[p[cur]]))
                sv_pre = sv_cur
        for key in result.keys():
            result[key] /= samples
            print("%d worker's shapley value: %.6f" % (key, result[key]))

    # def _shapley_value_sampling(self, d_params):
    #     """
    #     Calculate Shapley Values for clients
    #
    #     ARGS:
    #         d_params:
    #     RETURN:
    #         result(dict): Client weights' shapely valye
    #     """
    #     y = [d_params[i] for i in [2, 0, 1]]
    #     y_0 = [d_params[j] for j in [0, 1, 2]]
    #     print(type(y[0]))
    #     y_ag, y_0_ag = self._aggregate(y), self._aggregate(y_0)
    #     for k in y_ag.keys():
    #         print(y_ag[k] - y_0_ag[k])
    #     # u_y, u_y0 = self._evaluate(y_ag), self._evaluate(y_0_ag)
    #     # delta = u_y - u_y0
    #     # print(delta)