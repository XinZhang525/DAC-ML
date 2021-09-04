import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import itertools
import numpy as np
import struct
import math
import random
import wandb
from helper import *

region_width = 5
region_length = 5

def MSE_loss(y_test, y_pred):
    MSE = F.mse_loss(y_pred, y_test)
    return MSE

def test(vae, test_loader, task_length, task_iter, opt, max_num, device):
    vae.eval()
    test_loss = 0
    all_MAPE = 0
    for step, (speed, inflow, demand, sub_period) in enumerate(test_loader):
        ################# prepare input of vae ###############
        # prepare period in subtask, enlarge a number into an image
        period = torch.zeros(region_width, region_length).float() + sub_period[0].float()
        for i in range(1, task_length):
            tmp_pod = torch.zeros(region_width, region_length).float() + sub_period[i].float()
            period = torch.cat((period, tmp_pod), dim=0)
        period = period.view(task_length, 1, region_width, region_length).float()

        target, support1, support2 = target_identifier(opt.target, speed, inflow, demand)
        
        support1 = support1.view(task_length, 1, region_width, region_length).float()
        support2 = support2.view(task_length, 1, region_width, region_length).float()
        # prepare x_train
        x_train = torch.cat((support1, support2), dim=1)
        x_train = torch.cat((x_train, period), dim=1)

        # prepare y, select the center value of an image as the label, and enlarge a number into an image
        y = [target[0, int(region_width / 2), int(region_length / 2)]]
        for i in range(1, task_length):
            y_tmp = target[i, int(region_width / 2), int(region_length / 2)]
            y.append(y_tmp)
        y_train = torch.FloatTensor(y)
        y_test = y

        tmp_y_train = torch.zeros(region_width, region_length)
        for i in range(task_length):
            tmp = torch.zeros(region_width, region_length) + y_train[i]
            tmp_y_train = torch.cat((tmp_y_train, tmp), dim=0)
        y_train = tmp_y_train.view(task_length + 1, 1, region_width, region_length)
        
        d_h = Variable(torch.zeros(opt.D_out * opt.num_layers).to(device))
        t_h = Variable(torch.zeros(opt.D_out * opt.num_layers).to(device))
        for t in range(task_length):
            # prepare domain data
            domain_data = prepare_domain(x_train[:t+1], y_train[:t+1], opt.domains, device)
            # prepare xy_train
            x = x_train[t].view(1, -1, region_width, region_length)
                
            y = y_train[t].view(1, -1, region_width, region_length)
    
            xy_train = torch.cat((x, y), dim=1)    # shape: (1, 4, 5, 5)
            y_true = torch.FloatTensor([y_test[t]])
            
            xy_train = Variable(xy_train.to(device))
            y_true = Variable(y_true.to(device))
       
            y_pred, (d_mu, d_log_var, d_h), (t_mu, t_log_var, t_h) = vae(xy_train, domain_data, d_h, t_h) #  d_h, t_h
            y_pred = y_pred.cpu().data * max_num
            y_true = y_true.cpu().data * max_num
            loss = MSE_loss(y_true, y_pred)
            test_loss += loss.item()

            MAPE = np.absolute(y_pred.cpu().data.numpy() - y_true.cpu().data.numpy())/y_true.cpu().data.numpy()
            all_MAPE += MAPE[0]
            if opt.log_wandb and task_iter % 50 == 0:
                wandb.log({f"Epoch {task_iter} Test RMSE": math.sqrt(loss.item()), 
                           f"Epoch {task_iter} Test MAPE": MAPE,
                           "step": t})
            
#             print(f"Step {t} Test RMSE: ", math.sqrt(loss.item()))
#             print(f"Step {t} Test MAPE: ", MAPE)
        break
    return math.sqrt(test_loss/task_length), all_MAPE/task_length

def evaluate(vae, dataset, opt, task_iter, max_num, device):
    Sampler = task_sampler(dataset, opt.task_length)
    test_loader = Data.DataLoader(dataset=dataset, batch_size=opt.task_length, shuffle=False, sampler=Sampler)

    test_rmse, test_mape = test(vae, test_loader, opt.task_length, task_iter, opt, max_num, device)
    return test_rmse, test_mape
