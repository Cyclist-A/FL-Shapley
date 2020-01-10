"""
Two ways to control the training process:
    1. Pre-set total training epochs
    2. Set a threshold. Client automatically stop training while achieve the threshold

Explaination of some clients' settings:
    mode: 'epoch' or 'thres'
    epoch: Training epochs for each round. Default as 30. Only works when mode is 'epoch'
    thres: Threshold for training accuracy. Default as 0.8. Only works when mode is 'thres'
    max_epoch: Maximum training epoch in 'thres' mode, default 1e5. Only works when mode is 'thres'
    eval_each_iter: whether evaluate in each iteration. Only works when mode is 'thres'
    ... 
"""

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.distributions as distributions

from sklearn.metrics import accuracy_score


def random_response(params, std):
    """
    Use Normal distribution to generate 'trained weights' where mean is params from server, 
    and std is set by user
    
    ARGS:
        params: NN's params, used as mean in distribution
        std: distribution's std
    RETURN:
        random_weigt(dict): a state dict of NN gerenated randomly according to params from server
    """
    # generate random response according to params from server
    random_weight = {}
    for layer in params:
        normal = distributions.normal.Normal(params[layer].to(torch.float), std)
        random_weight[layer] = normal.sample()
    
    return random_weight


def run(clients, client_net, net_kwargs, params, device, channel, settings):
    """
    Train the model several epochs locally

    AGRS:
        clients: clients' datasets. They are assigned to train on this device
        client_net: NN used to train
        net_kwargs: args to create the NN
        params: NN's parameters
        device: the torch device runs this NN
        channel: (channel_in, channel_out): communication channel and controller
        settings: training setttings
    RETURN:
        None
    """
    # initialize before training
    if net_kwargs:
        net = client_net(**net_kwargs).to(device, torch.float64)
    else:
        net = client_net().to(device, torch.float64)

    # run for each client
    for idx in clients:
        # initialize
        net.load_state_dict(params)
        loader = utils.DataLoader(clients[idx], batch_size=settings['batch_size'], shuffle=True, pin_memory=True, num_workers=5)
        test_loader = utils.DataLoader(clients[idx], batch_size=2000, shuffle=False, pin_memory=True, num_workers=10)
        criterion = settings['loss_func']()
        optimizer = settings['optimizer'](net.parameters(), lr=settings['lr'])

        if settings['enable_scheduler']:
            scheduler = settings['scheduler'](optimizer, **settings['scheduler_settings'])

        if settings['mode'] == 'epoch':
            total_epoch = settings['epoch']
        elif settings['mode'] == 'thres':
            total_epoch = settings['max_epoch']
        else:
            raise ValueError(f"Invaild training mode. Can only be 'epoch' or 'thres'. Got {settings['mode']}")
        
        total_iter = 0
        stop_flg = False

        # start training
        for epoch in range(total_epoch):
            net.train()
            epoch_loss = 0

            for i, data in enumerate(loader, 1):
                total_iter += 1

                inputs, labels = data[0].to(device, torch.float64), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
                if settings['mode'] == 'thres' and settings['eval_each_iter']:
                    train_accu = _evaluate(net, test_loader, device)
                    net.train()

                    if train_accu >= settings['thres']:
                        stop_flg = True
                        break
            
            # stop
            if stop_flg:
                break

            # scheduler
            if settings['enable_scheduler']:
                scheduler.step(epoch_loss)

            # evaluate
            train_accu = _evaluate(net, test_loader, device)
            if settings['mode'] == 'thres' and train_accu >= settings['thres']:
                break
            
            if settings['verbose']:
                if settings['mode'] == 'thres':
                    print(f'Client {idx}: Epoch[{epoch + 1}] Loss: {epoch_loss/i} | Train accu: {train_accu}')
                else:
                    print(f'Client {idx}: Epoch[{epoch + 1}/{settings["epoch"]}] Loss: {epoch_loss/i} | Train accu: {train_accu}')
                sys.stdout.flush()
        
        # finished, send params back to server
        trained_params = net.state_dict()   
        message = {
            'id': idx,
            'params': {layer: trained_params[layer].clone().detach().cpu() for layer in trained_params},
            'length': len(clients[idx]),
            'iter': total_iter
        }
        channel[1].put(message)
    
    # wait for exit signal from server
    channel[0].get()

def _evaluate(net, loader, device):
    """
    Evaluate current net in the server

    ARGS:
        net: NN for evaluation
        loader: dataloader for evaluation
        device: device to run the NN
    RETURN:
        accuracy_score: evalutaion result
    """
    # initialize
    predicted = []
    truth = []

    net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device, torch.float64), data[1].to(device)
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)

            for p, q in zip(pred, labels):
                predicted.append(p.item())
                truth.append(q.item())

    return accuracy_score(truth, predicted)