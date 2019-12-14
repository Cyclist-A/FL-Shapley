import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.multiprocessing as mp

from sklearn.metrics import accuracy_score

import client
from splitDataset import split_dataset
from valuation import shapley_value, leave_one_out

class FederatedServer:
    """
    A Framework for federated learning. Provide a for-loop version and a multiprocessing
    version. The multiprocessing version can only run on LinuxOS. For multiprocessing,
    it average clients to every GPU
    
    ARGS:
        net: a pytorch neural network class
        trainset: the whole training set, split in i.i.d
        testset: testset to evaluate server's preformance
        net_kwargs: args to create the NN
        client_settings: settings for training clients
        devices: a list of available devices
        cal_sv: whether to calculate Shapley Value
        cal_loo: whether to calculate LOO
        clients_num: # clients to create
        split_method: the method to split the dataset. 'iid', 'imba-label', 'imba-size'
        imbalanced_rate: imbalance strength, only works when split method is imba-label
        capacity: capacity for each client, only works when split method is imba-size
        warm_up: whether to warm up the server
        warm_setting: setting for training whem warm up
        random_response: whether to create a client with random weights' response
        std: .Only works when random response is true
        random_state: set the seed to split dataset
    """
    def __init__(self, net, trainset, testset, net_kwargs=None, client_settings=None, devices=['cpu'], cal_sv = True, cal_loo = True,
        clients_num=3, split_method='iid', imbalanced_rate=0.8, capacity=[0.1, 0.3, 0.6], warm_up=False, warm_setting=None, random_response=False, std=0.5, random_state=100):
        # initialize
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.net_kwargs = net_kwargs
        self.devices = [torch.device(d) for d in devices]
        self.cal_sv = cal_sv
        self.cal_loo = cal_loo
        self.clients_num = clients_num
        self.random_response = random_response
        self.std = 0.5

        if net_kwargs:
            self.current_params = net(**self.net_kwargs).state_dict()
        else:
            self.current_params = net().state_dict()

        # fix random state (BUG cannot reproduce)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # split dataset for different clients
        self.clients = split_dataset(trainset, self.clients_num, split_method, imbalanced_rate, capacity)

        # construct clients info
        self.client_settings = {
            'epoch': 3,
            'lr': 0.01,
            'batch_size': 128,
            'loss_func': nn.CrossEntropyLoss,
            'optimizer': optim.Adam,
            'verbose': True
        }

        if client_settings:
            for k in client_settings:
                if k in self.client_settings:
                    self.client_settings[k] = client_settings[k]
                else:
                    raise RuntimeError('Wrong settings for client, no such settings')

        # warm up the server
        if warm_up:
            self._warm_up(warm_setting)

    def run(self, rounds=3, c=1):
        """
        Run federated for several rounds

        ARGS:
            rounds: the number of rounds to run federated learning
            c: the proportion of chosen clients in each rounds
        RETURN:
            None
        """
        for r in range(rounds):
            # choose clients
            clients_idx = self._choose_clients(c)

            # train on clients
            params = self._train_clients(clients_idx)

            # valuation
            if self.cal_sv:
                shapley_value(self.net, self.net_kwargs, self.testset, params, self.devices)
            if self.cal_loo:
                leave_one_out(self.net, self.net_kwargs, self.testset, params, self.devices[0])

            # update params in server
            self._step(params)
            
            # evaluate
            test_accu = self._evaluate()
            print(f"Rounds[{r+1}/{rounds}]: Test Accu: {test_accu}")
            print('-' * 20)

    def _warm_up(self, settings):
        """
        Train server before starting client processes

        ARGS:
            settings: customed settings to warm up
        RETURN:
            None
        """
        default_setting = {
            'warm_rate': 0.01,
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

        # create warm set
        idx = [i for i in range(len(self.trainset))]
        np.random.shuffle(idx)
        warm_idx = idx[:int(len(self.trainset) * default_setting['warm_rate'])]
        warm_set = utils.Subset(self.trainset, warm_idx)

        # initialize before training
        device = self.devices[0]
        if self.net_kwargs:
            net = self.net(**self.net_kwargs).to(device)
        else:
            net = self.net().to(device)

        loader = utils.DataLoader(warm_set, batch_size=default_setting['batch_size'] , shuffle=True)
        criterion = default_setting['loss_func']()
        optimizer = default_setting['optimizer'](net.parameters(), lr=default_setting['lr'])

        print('Start heating server...')

        for epoch in range(default_setting['epoch']):
            net.train()
            epoch_loss = 0

            for i, data in enumerate(loader, 1):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # updata
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            # evaluate
            test_accu = self._evaluate()
            print(f'Epoch[{epoch+1}/{default_setting["epoch"]}] Loss: {epoch_loss/i} | Test accu: {test_accu}')
        
        # store the training result
        params = net.state_dict()
        self.current_params = {}
        for k in params:
            self.current_params[k] = params[k].clone().detach().cpu()

        print('Finished heating server.')       
    
    def _choose_clients(self, c):
        """
        Randomly choose some clients to update weight
        
        ARGS:
            c: the proportion of chosen clients, should lie in (0, 1]
        RETURN:
            idx(list): client id which are chosen
        """
        # select a proportion of clients
        if c > 1 or c <= 0:
            raise ValueError(f'The proportion of chosen clients should lie in (0, 1]. Now is {c}')
        
        # randomly choose clients idx
        num = max(c * self.clients_num, 1)
        idx = [i for i in range(self.clients_num)]
        np.random.shuffle(idx)
        idx = idx[:num]

        return idx

    def _train_clients(self, clients_idx):
        """
        Train clietns for a round
        
        ARGS:
            clients_idx: chosen clients id
        RETURN:
            params(dict): client_id: params; parameters from each client's training result
        """
        # distribute missions
        distri = [{} for i in range(min(len(self.devices), len(clients_idx)))]
        for i, idx in enumerate(clients_idx):
            distri[i%len(distri)][idx] = self.clients[idx]

        # build channel for every process
        channel_out = mp.Queue()
        channel = [(mp.Queue(), channel_out) for _ in distri]

        # start training locally
        print('Starting client processes...')
        clients_pro = [mp.Process(target=client.run, args=(distri[i], self.net, self.net_kwargs, copy.deepcopy(self.current_params), self.devices[i], channel[i], self.client_settings)) for i in range(len(distri))]
        for c in clients_pro:
            c.start()
        
        # collect results from clients pro
        params = {}
        length = {}
        while len(params) < len(clients_idx):
            message = channel_out.get()
            params[message['id']] = message['params']
            length[message['id']] = message['length']

        # get all params, kill the clients
        for c in channel:
            c[0].put(-1)

        # make sure all clients have finished
        for c in clients_pro:
            c.join()

         # add random response in every round
        if self.random_response:
            params['random'] = client.random_response(copy.deepcopy(self.current_params), self.std)
            # use ave length as random's length
            length['random'] = sum(length.values()) / len(clients_idx)

        # scale the params
        total = sum(length.values())
        for idx in params:
            for layer in params[idx]:
                params[idx][layer] = torch.div(params[idx][layer], total / length[idx])

        return params

    def _step(self, params):
        """
        Aggregate delta parameters from different clients.
        Use aggregation resutls to update the model

        ARGS:
            params: all parameters from clients
            length: client datasets' lenght
        RETURN:
            None
        """
        idx = list(params.keys())

        # aggregate
        layers = params[idx[0]].keys()
        aggr_params = {}

        for l in layers:
            aggr_params[l] = params[idx[0]][l].clone().detach()
            for i in idx[1:]:
                aggr_params[l] += params[i][l]

        # update net's params
        self.current_params = aggr_params

    def _evaluate(self):
        """
        Evaluate current net in the server

        ARGS:
            None
        RETURN:
            accuracy_score: evalutaion result
        """
        # initialize
        device = self.devices[-1]
        if self.net_kwargs:
            net = self.net(**self.net_kwargs).to(device)
        else:
            net = self.net().to(device)
        net.load_state_dict(self.current_params)
        loader = utils.DataLoader(self.testset, batch_size=2000, shuffle=False, num_workers=5)
        predicted = []
        truth = []

        net.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, pred = torch.max(outputs.data, 1)

                for p, q in zip(pred, labels):
                    predicted.append(p.item())
                    truth.append(q.item())

        return accuracy_score(truth, predicted)
