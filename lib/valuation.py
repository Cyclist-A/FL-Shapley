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


def eval_each_client(net, net_kwargs, dataset, params, device):
    """
    Evaluate clients parameters on testset

    ARGS:
        net: trained NN
        net_kwargs: kwargs for creating net
        dataset: evaluation dataset
        params: params uploaded from clients
    RETURN:
        result(dict): Each client weight's LOO evaluation value
    """
    print('Start evaluating parameters from clients...')
    if net_kwargs:
        net = net(**net_kwargs)
    else:
        net = net()

    device = torch.device(device)
    loader = utils.DataLoader(dataset, batch_size=2000, shuffle=False, pin_memory=True, num_workers=10)
    res = {}
    for k in params:
        net.load_state_dict(params[k])
        res[k] = _evaluate(net, loader, device)
    
    return res

def leave_one_out(net, net_kwargs, dataset, params, scale, device):
    """
    Calculate Leave-One-Out(LOO) evaluation

    ARGS:
        net: trained NN
        net_kwargs: kwargs for creating net
        dataset: evaluation dataset
        params: params uploaded from clients
        scale: the scale value for each client's params
    RETURN:
        result(dict): Each client weight's LOO evaluation value
    """
    print('Start calculating LOO values...')
    client_ids = params.keys()
    result = defaultdict(float)
    
    if net_kwargs:
        net = net(**net_kwargs)
    else:
        net = net()

    net.load_state_dict(_aggregate(params, scale))
    loader = utils.DataLoader(dataset, batch_size=5000, shuffle=False, pin_memory=True)
    global_result = _evaluate(net, loader, device)

    for c in client_ids:
        cur_params = _aggregate({idx: params[idx] for idx in client_ids if idx != c}, scale)
        net.load_state_dict(cur_params)
        res = global_result - _evaluate(net, loader, device)
        result[c] = res
    
    # print results
    for key in result.keys():
        print("%d worker's LOO value: %.6f" % (key, result[key]))
    
    return result

def shapley_value(net, net_kwargs, dataset, params, scale, devices):
    """
    Implement shapley value using multiprocessing
    ARGS:
        net: net for evaluation
        net_kwargs: kwargs for creating net
        dataset: dataset for evaluation
        params: params of all chosen clients
        scale: the scale value for each client's params
        devices: available devices list
    RETURN:
        result(dict): Each client weight's SV evaluation value
    """
    # initialize
    N = len(params)
    w_ids = list(params.keys())
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
                                 args=(net, net_kwargs, dataset, params, scale, tasks[i], samples, devices[i], channel_out))
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

    return sv

def _shapley_value(net, net_kwargs, dataset, params, scale, permutations, samples, device, channel_out):
    """
    Evaluate multi differernt params of a NN

    ARGS:
        net: training NN
        net_kwargs: kwargs for creating net
        dataset: evaluation dataset
        params: params dict from chosn clients
        scale: the scale value for each client's params
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
        pre_sv = 0.0
        for cur in range(len(params)):
            cur_params = _aggregate({idx: params[idx] for idx in p[:cur + 1]}, scale)
            net.load_state_dict(cur_params)
            cur_sv = _evaluate(net, loader, device)
            res[p[cur]] += (cur_sv - pre_sv) / samples
            pre_sv = cur_sv

    channel_out.put(res)

def _aggregate(params, scale):
    """
    Aggregate params from different NN

    ARGS:
        params: a list contains workers' weights
        scale: the scale value for each client's params
    RETURN:
        aggr_params(dict): a aggregated weights
    """
    if not params:
        return {}

    clients = list(params.keys())
    # calcualte new scale by clients
    total = 0
    for c in clients:
        total += scale[c]
    new_scale = {c:scale[c] / total for c in clients}
    
    aggr_params = {}
    layers = params[clients[0]].keys()

    for l in layers:
        aggr_params[l] = params[clients[0]][l].clone().detach() * new_scale[clients[0]]
        for i in clients[1:]:
            aggr_params[l] += params[i][l] * new_scale[i]

    return aggr_params


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
