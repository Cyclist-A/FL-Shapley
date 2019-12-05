from server import Server
from client import Client

import numpy as np
import math
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
        trainset: the whole training set, split in i.i.d
        testset: testset to evaluate server's preformance
        devices: a list of available devices
        C: # clients to create
        split_method: the method to split the dataset. 'iid', 'imba-label', 'imba-size'
        imbalanced_rate: imbalance strength, only works when split method is imba-label
        capacity: capacity for each client, only works when split method is imba-size
        warm_rate: the proportion of trainset owned by server, only works when server needs warm up
        random_state: set the seed to split dataset
    """
    def __init__(self, net, trainset, testset, devices='cpu', C=3, split_method='iid', imbalanced_rate=0.8, 
        capacity=[0.1, 0.3, 0.6], warm_rate=0.01, random_state=100):
        # construct channel to connect server and client
        self.C = C
        channel_server_in = [mp.Queue() for i in range(C)]
        channel_server_out = [mp.Queue() for i in range(C)]

        # split dataset for different clients
        subsets, warm_set = self._split_dataset(trainset, split_method, imbalanced_rate, capacity, warm_rate, random_state)

        # create a server and clients
        self.server = Server(net(), channel_server_in, channel_server_out, testset, warm_set, device=devices[0])
        self.clients = [Client(i, net(), channel_server_out[i], channel_server_in[i], subsets[i], devices[i%len(devices)]) for i in range(C)]

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
            weights = {j: c.run_round(current_param) for j, c in enumerate(self.clients)}

            # calculate shapley value            
            # self.server._shapley_value_sampling(weights)

            # calculate LOO value
            # self.server._leave_one_out(weights)

            # aggregate the paramters
            current_param = weights[0]
            for k in current_param:
                for i in range(1, len(weights)):                
                    current_param[k] += weights[i][k]
                current_param[k] = torch.div(current_param[k], len(weights))
            
            # evaluate
            accu = self.server._evaluate(current_param)
            print(f'Round[{i+1}/{rounds}] Test Accu: {accu}')


    def _split_dataset(self, dataset, split_method, imbalanced_rate, capacity, warm_rate, random_state):
        """
        Split dataset by assigned different datasets for each clients
        
        ARGS:
            dataset: dataset to be splited
            split_method: whether to sample in non-iid way
            imbalanced_rate: imbalance strength, only works when split method is imba-label
            capacity: capacity for each client, only works when split method is imba-size
            warm_rate: the proportion of trainset owned by server, only works when server needs warm up 
            ramdom_state: control random seed
        RETURN:
            subsets(list): A list of datasets for clients
            warm_set(utils.dataset): Use for warm up training in server
        """
        # set random state
        np.random.seed(random_state)      

        # split by different methods
        if split_method == 'imba-label':
            subset_idx = self._split_imba_label(dataset, imbalanced_rate)
        
        elif split_method == 'imba-size':
            subset_idx = self._split_imba_size(dataset, capacity)

        elif split_method == 'iid':
            subset_idx = self._split_iid(dataset)

        else:
            raise ValueError(f"Split method can only be 'iid', 'imba-label' or 'imba-size'. \
                Your input is {split_method} ")

        subsets = [utils.Subset(dataset, i) for i in subset_idx]

        # create warm set
        idx = [i for i in range(len(dataset))]
        np.random.shuffle(idx)
        warm_idx = idx[:int(len(dataset) * warm_rate)]
        warm_set = utils.Subset(dataset, warm_idx)

        return subsets, warm_set     

    def _split_imba_label(self, dataset, imbalanced_rate):
        """
        Randomly choose a label, and assign most of its training points to the last client.
        All subset 
        BUG: if the number of the label's data points is larger then the client's capacity,
            it'll raise error

        ARGS:
            dataset: the dataset need to be splited
            imbalanced_rate: the proportion of 
        RETURN:
            subset_idx(list): a list contains each subset data's index
        """
        subset_idx = [[] for i in range(self.C)]

        # randomly choose a label to become non_iid
        label = dataset[np.random.randint(len(dataset))][1]

        # split idx by label (whether chosen)
        normal_idx = []
        label_idx = []
        for i, data in enumerate(dataset):
            if data[1] == label:
                label_idx.append(i)
            else:
                normal_idx.append(i)
        np.random.shuffle(normal_idx)
        np.random.shuffle(label_idx)

        # construct subset list, last clients will have most choosen label
        # split data by split point
        label_split_point = int(len(label_idx) * imbalanced_rate)
        normal_split_point = int(len(dataset) / self.C) - label_split_point

        # bug detection
        if normal_split_point <= 0:
            raise RuntimeError("Overloaded. The number of data points isover than client's capacity.")

        subset_idx[-1] += label_idx[:label_split_point]
        subset_idx[-1] += normal_idx[:normal_split_point]

        # remain index for other clients
        remain_idx = normal_idx[normal_split_point:] + label_idx[label_split_point:]
        for i, idx in enumerate(remain_idx):
            subset_idx[i % (self.C-1)].append(idx)

        return subset_idx

    def _split_imba_size(self, dataset, capacity):
        """
        ARGS:
            dataset: the dataset need to be splited
            capacity: capacity for each client, only works when split method is imba-size
        RETURN:
            subset_idx(list): a list contains each subset data's index
        """
        # check if capacity is valid
        if len(capacity) != self.C:
            raise ValueError(f'The length of capacity(got {len(capacity)}) should be same as the number of clients(got{self.C})')
        elif abs(sum(capacity)) - 1 > 1e-8:
            raise ValueError(f'The sum of the capacity list should be 1, got{sum(capacity)}')
        
        split_idx = [int(i * len(dataset)) for i in capacity[:-1]]
        split_idx.insert(0, 0)
        split_idx.append(len(dataset))

        idx = [i for i in range(len(dataset))]
        np.random.shuffle(idx)
        subset_idx = [idx[split_idx[i]:split_idx[i+1]] for i in range(self.C)]

        return subset_idx


    def _split_iid(self, dataset):
        """
        Randomly split dataset in i.i.d
        
        ARGS:
            dataset: the dataset need to be splited
        RETURN:
            subset_idx(list): a list contains each subset data's index
        """
        subset_idx = [[] for i in range(self.C)]
        tmp = [i for i in range(len(dataset))]
        np.random.shuffle(tmp)
        for i, t in enumerate(tmp):
            subset_idx[i%self.C].append(t)
        
        return subset_idx
