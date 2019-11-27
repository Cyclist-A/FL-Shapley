from server import Server
from client import Client

import numpy as np
import multiprocessing as mp

import torch
import torch.utils.data as utils

class Federated:
    """
    A Framework for federated learning
    
    ARGS:
        net: a pytorch neural network class 
        C: # clients to create
        trainset: the whole training set, split in i.i.d
        testset: testset to evaluate server's preformance
        random_state: set the seed to split dataset
    """
    def __init__(self, net, C, trainest, testset, random_state=100):
        # construct channel to connect server and client
        self.C = C
        channel = [mp.Queue() for i in range(C)]

        # split dataset for different clients
        samplers, warmer = self._split_dataset(trainest, random_state)

        # create a server and clients
        self.server = Server(net, channel, testset, trainest, warmer)
        self.clients = [Client(net, channel[i], trainest[i]) for i in range(C)]

    def run(self, batch_size=32, warm_up=True):
        """
        TODO
        """
        
        
    def _split_dataset(self, dataset, random_state):
        """
        Split dataset by assigned different samplers for each clients
        
        ARGS:
            dataset: dataset to be splited
            ramdom_state: control random seed
        RETURN:
            samplers(list): A list of samplers for clients
            warm_sampler(utils.Sampler): Use for warm up training in server
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

        samplers = [utils.SubsetRandomSampler(i) for i in idx]
        warm_sampler = utils.SubsetRandomSampler(warm_idx)

        return samplers, warm_sampler
        