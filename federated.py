from server import Server
from client import Client

import numpy as np
import torch.multiprocessing as mp

import torch
import torch.utils.data as utils

class Federated:
    """
    A Framework for federated learning. Provide a for-loop version and a multiprocessing
    version. The multiprocessing version can only run on LinuxOS. For multiprocessing, 
    it average clients to every 
    
    ARGS:
        net: a pytorch neural network class 
        C: # clients to create
        trainset: the whole training set, split in i.i.d
        testset: testset to evaluate server's preformance
        devices: a list of available devices
        random_state: set the seed to split dataset
    """
    def __init__(self, net, C, trainest, testset, devices, random_state=100):
        # construct channel to connect server and client
        self.C = C
        channel = [mp.Queue() for i in range(C)]

        # split dataset for different clients
        subsets, warm_set = self._split_dataset(trainest, random_state)

        # create a server and clients
        self.server = Server(net(), channel, testset, warm_set, device=devices[0])
        self.clients = [Client(net(), channel[i], subsets[i], devices[i%len(devices)]) for i in range(C)]

    def run(self, server_settings={}, client_settings=None):
        """
        Fork a process for each clients and the server
        Run their main function
        
        ARGS:
            server_settings(optional): settings for running the server
            client_settings(optional): settings for running the client
        RETURN:
            None
        """
        # start server
        print('Starting server process...')
        server_pro = mp.Process(target=self.server.run, kwargs=server_settings)
        server_pro.start()

        # assign processes to clients
        print('Starting client processes...')
        clients_pro = [mp.Process(target=c.run, args=(client_settings, )) for c in self.clients]
        for c in clients_pro:
            c.start()
        
        # wait until all finished
        for c in clients_pro:
            c.join()
        print('All clients have exited')

        server_pro.join()
        print('Server stopped working')
    
    def run_for_loop(self, rounds=3):
        """
        Use for loop to run FL instead of multiprocessing
        In this situation, this class works as a server
        
        RETURN:
            None
        """
        # initialize
        # self.server._warm_up(None)
        current_param = self.server.net.state_dict()

        for i in range(rounds):
            # choose clients

            # run clients
            weights = {i: c.run_round(current_param) for i, c in enumerate(self.clients)}

            # calculate shapley value
            # self.server._shapley_value_sampling(weights)

            # aggregate the paramters
            current_param = weights[0]
            for k in current_param:
                for i in range(1, len(weights)):                
                    current_param[k] += weights[i][k]
                current_param[k] /= len(weights)

            # evaluate
            accu = self.server._evaluate(current_param)
            print(f'Round[{i+1}/{rounds}] Test Accu: {accu}')


    def _split_dataset(self, dataset, random_state):
        """
        Split dataset by assigned different datasets for each clients
        
        ARGS:
            dataset: dataset to be splited
            ramdom_state: control random seed
        RETURN:
            subsets(list): A list of datasets for clients
            warm_set(utils.dataset): Use for warm up training in server
        """
        torch.manual_seed(random_state)
        
        # construct subset list
        idx = [[] for i in range(self.C)]

        tmp = [i for i in range(len(dataset))]
        np.random.shuffle(tmp)
        for i, t in enumerate(tmp):
            idx[i%self.C].append(t)

        # build warm up idx
        warm_idx = []
        for l in idx:
            for i, t in enumerate(l):
                if i % self.C == 0:
                    warm_idx.append(i)

        subsets = [utils.Subset(dataset, i) for i in idx]
        warm_set = utils.Subset(dataset, warm_idx)

        return subsets, warm_set
        