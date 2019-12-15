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
        net_kwargs: kargs for creating NN
        params: NN's parameters
        device: the torch device runs this NN
        channel: (channel_in, channel_out): communication channel and controller
        settings: training setttings
    RETURN:
        None
    """
    # initialize before training
    if net_kwargs:
        net = client_net(**net_kwargs).to(device)
    else:
        net = client_net().to(device)

    # run for each client
    for idx in clients:
        net.load_state_dict(params)
        loader = utils.DataLoader(clients[idx], batch_size=settings['batch_size'], shuffle=True, pin_memory=True, num_workers=5)
        criterion = settings['loss_func']()
        optimizer = settings['optimizer'](net.parameters(), lr=settings['lr'])

        # start training
        for epoch in range(settings['epoch']):
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
            train_accu = _evaluate(net, clients[idx], device)
            if settings['verbose']:
                print(f'Client {idx}: Epoch[{epoch + 1}/{settings["epoch"]}] Loss: {epoch_loss/i} | Train accu: {train_accu}')
            sys.stdout.flush()
        
        # finished, send params back to server
        trained_params = net.state_dict()
        message = {
            'id': idx,
            'params': {layer: trained_params[layer].clone().detach().cpu() for layer in trained_params},
            'length': len(clients[idx])
        }
        channel[1].put(message)
    
    # wait for exit signal from server
    channel[0].get()

def _evaluate(net, dataset, device):
    """
    Evaluate current net in the server

    ARGS:
        net: NN for evaluation
        dataset: dataset for evaluation
        device: device to run the NN
    RETURN:
        accuracy_score: evalutaion result
    """
    # initialize
    net.to(device)
    loader = utils.DataLoader(dataset, batch_size=5000, shuffle=False)
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