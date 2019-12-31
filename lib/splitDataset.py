"""
Training set needs to be split for each client in federated learning
'split_dataset' provides three different ways to split training set
"""

import numpy as np
import torch.utils.data as utils


def split_dataset(dataset, clients_num, split_method, imbalanced_rate, capacity):
    """
    Split dataset by assigned different datasets for each clients
    
    ARGS:
        dataset: dataset to be splited
        clients_num: the number of clients
        split_method: whether to sample in non-iid way
        imbalanced_rate: imbalance strength, only works when split method is imba-label
        capacity: capacity for each client, only works when split method is imba-size
    RETURN:
        subsets(dict): client_id: subset; the subset for each clients
    """
    # split by different methods
    if split_method == 'imba-label':
        subset_idx = _split_imba_label(dataset, clients_num, imbalanced_rate)
    
    elif split_method == 'imba-size':
        subset_idx = _split_imba_size(dataset, clients_num, capacity)

    elif split_method == 'iid':
        subset_idx = _split_iid(dataset, clients_num)

    else:
        raise ValueError(f"Split method can only be 'iid', 'imba-label' or 'imba-size'. \
            Your input is {split_method} ")

    subsets = {i: utils.Subset(dataset, s) for i, s in enumerate(subset_idx)}

    return subsets


def _split_imba_label(dataset, clients_num, imbalanced_rate):
    """
    Randomly choose a label, and assign most of its training points to the last client.
    All subset 
    BUG: if the number of the label's data points is larger then the client's capacity,
        it'll raise error

    ARGS:
        dataset: the dataset need to be splited
        clients_num: the number of clients
        imbalanced_rate: the proportion of 
    RETURN:
        subset_idx(list): a list contains each subset data's index
    """
    subset_idx = [[] for i in range(clients_num)]

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
    normal_split_point = int(len(dataset) / clients_num) - label_split_point

    # bug detection
    if normal_split_point <= 0:
        raise RuntimeError("Overloaded. The number of data points isover than client's capacity.")

    subset_idx[-1] += label_idx[:label_split_point]
    subset_idx[-1] += normal_idx[:normal_split_point]

    # remain index for other clients
    remain_idx = normal_idx[normal_split_point:] + label_idx[label_split_point:]
    for i, idx in enumerate(remain_idx):
        subset_idx[i % (clients_num-1)].append(idx)

    return subset_idx


def _split_imba_size(dataset, clients_num, capacity):
    """
    ARGS:
        dataset: the dataset need to be splited
        clients_num: the number of clients
        capacity: capacity for each client, only works when split method is imba-size
    RETURN:
        subset_idx(list): a list contains each subset data's index
    """
    # check if capacity is valid
    if len(capacity) != clients_num:
        raise ValueError(f'The length of capacity(got {len(capacity)}) should be same as the number of clients(got{clients_num})')
    elif abs(sum(capacity)) - 1 > 1e-8:
        raise ValueError(f'The sum of the capacity list should be 1, got{sum(capacity)}')
    
    split_idx = [int(i * len(dataset)) for i in capacity[:-1]]
    split_idx.insert(0, 0)
    split_idx.append(len(dataset))

    idx = [i for i in range(len(dataset))]
    np.random.shuffle(idx)
    subset_idx = [idx[split_idx[i]:split_idx[i+1]] for i in range(clients_num)]

    return subset_idx


def _split_iid(dataset, clients_num):
    """
    Randomly split dataset in i.i.d
    
    ARGS:
        dataset: the dataset need to be splited
        clients_num: the number of clients
    RETURN:
        subset_idx(list): a list contains each subset data's index
    """
    subset_idx = [[] for i in range(clients_num)]
    tmp = [i for i in range(len(dataset))]
    np.random.shuffle(tmp)
    for i, t in enumerate(tmp):
        subset_idx[i%clients_num].append(t)
    
    return subset_idx