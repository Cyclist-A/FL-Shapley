"""
This file implement data valuation parts
"""
import math
import copy
import json
import itertools
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torch.utils.data as utils

import numpy as np
from sklearn.metrics import accuracy_score


def leave_one_out(net, net_kwargs, dataset, weights, device):
    """
    Calculate Leave-One-Out(LOO) evaluation

    ARGS:
        net: trained NN
        net_kwargs: kwargs for creating net
        dataset: evaluation dataset
        weights: weights uploaded from clients
    RETURN:
        result(dict): Each client's weight's LOO evaluation value
    """
    w_ids = weights.keys()
    result = defaultdict(float)
    
    if net_kwargs:
        net = net(**net_kwargs)
    else:
        net = net()

    net.load_state_dict(_aggregate(weights))
    loader = utils.DataLoader(dataset, batch_size=5000, shuffle=False, pin_memory=True)
    global_result = _evaluate(net, loader, device)

    for w in w_ids:
        print("evaluating %d weight's LOO value..." % w)
        cur_weight = _aggregate([weights[wk_id] for wk_id in w_ids if wk_id != w])
        net.load_state_dict(cur_weight)
        res = global_result - _evaluate(net, loader, device)
        result[w] = res
    for key in result.keys():
        print("%d worker's LOO value: %.6f" % (key, result[key]))

def shapley_value(net, net_kwargs, dataset, weights, devices):
    """
    Implement shapley value using multiprocessing
    ARGS:
        net: net for evaluation
        net_kwargs: kwargs for creating net
        dataset: dataset for evaluation
        weights: weights of all chosen clients
        devices: available devices list
    """
    # initialize
    N = len(weights)
    w_ids = list(weights.keys())
    print("Start to calculate shapley values...")

    # assign tasks and samples
    tasks = [[] for i in range(len(devices))]
    if N < 5:  # brute-force all possibilities
        samples = math.factorial(N)
        for i, p in enumerate(itertools.permutations(w_ids, N)):
            tasks[i%len(devices)].append(p)
    else:  # calculate by sampling
        samples = min(int(math.sqrt(math.factorial(N)) * 10), 1000)
        for i in range(samples):
            tasks[i%len(devices)].append(np.random.permutation(w_ids))

    # assign evaluation task to every device
    channel_out = mp.Queue()
    evaluation_pro = [mp.Process(target=_shapley_value,
                                 args=(net, net_kwargs, dataset, weights, tasks[i], samples, devices[i], channel_out))
                      for i in range(len(tasks))]
    
    for p in evaluation_pro:
        p.start()

    # get outputs
    sv = defaultdict(float)
    for i in range(len(tasks)):
        res = channel_out.get()
        for r in res:
            sv[r] += res[r]
    
    # make sure all clients left
    for p in evaluation_pro:
        p.join()

    for k in sorted(list(sv.keys())):
        print(f"Client {k}'s SV is: {sv[k]}")


def _shapley_value(net, net_kwargs, dataset, weights, permutations, samples, device, channel_out):
    """
    Evaluate multi differernt weights of a NN

    ARGS:
        net: training NN
        net_kwargs: kwargs for creating net
        dataset: evaluation dataset
        weights: weight dict from chosn clients
        permutation: the permutation assgined to calculate SV
        samples: total sample times
        device: torch device to run NN
        channel_out: channel to return SV to parent process
    RETURN:
        None
    """
    if net_kwargs:
        net = net(**net_kwargs)
    else:
        net = net()
    res = defaultdict(float)
    # ping loader to RAM
    loader = utils.DataLoader(dataset, batch_size=2000, shuffle=False, pin_memory=True, num_workers=5)

    # modified from original SV calculation
    for p in permutations:
        sv_pre = 0.0
        for cur in range(len(weights)):
            weight_cur = _aggregate([weights[wk_id] for wk_id in p[:cur + 1]])
            net.load_state_dict(weight_cur)
            sv_cur = _evaluate(net, loader, device)
            res[p[cur]] += (sv_cur - sv_pre) / samples
            sv_pre = sv_cur

    channel_out.put(res)

def _aggregate(weights):
    """
    Aggregate weights(after scaling) from different NN

    ARGS:
        weights:  a list contains workers' weights
    RETURN:
        aggr_params(dict): a aggregated weights
    """
    if not weights:
        return {}

    aggr_p = copy.deepcopy(weights[0])

    for k in weights[0].keys():
        for i in range(1, len(weights)):
            aggr_p[k] += weights[i][k]

    return aggr_p


def _evaluate(net, loader, device):
    # evaluate cur_weight
    predicted = []
    truth = []

    net = net.to(device)
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


def _utility(net, loader, device, alpha=0.5):
    # evaluate cur_weight
    predicted = []
    truth = []

    net = net.to(device)
    net.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    loss = 0.

    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            # print(outputs.shape, labels.shape, outputs[0], labels[0])
            _, pred = torch.max(outputs.data, 1)

            loss += loss_func(outputs, labels)

            for p, q in zip(pred, labels):
                predicted.append(p.item())
                truth.append(q.item())

    error = 1. - accuracy_score(truth, predicted)
    return loss * alpha + error * (1. - alpha)
