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
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import math
import struct
import random
import seaborn as sns
from VAE import Encoder, Encoder2
from VAE import Decoder
from VAE import DVAE2
from helper import *
from cbml_dmeval import *
import wandb
from datetime import datetime
from pathlib import Path
from argument import *

opt = get_args()

tags = ['Meta Learning', 'Traffic_prediction', f'Rigion({opt.region_i},{opt.region_j})']
domain_list = "_".join(opt.domains)
if opt.log_wandb:
    wandb.init(name=f"({opt.region_i},{opt.region_j})_{domain_list}",
               project=f"CBML_DM_Train_{opt.target}",
               tags=tags)
    wandb.config.update(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
cuda = True if torch.cuda.is_available() else False

datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_dir = Path(f'~/CBML/CBML_DM_{opt.target}_models') / domain_list / f'{datetime_now}'
ckpt_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
    
################################ Data Preprocessing ##########################################
# select the region and define period
(spd, inf, dmd), num_day, period_per_day, period = preprocess_data(opt)

region_width = 5
region_length = 5
if opt.target == 'speed':
    max_num = spd.max()
elif opt.target == 'inflow':
    max_num = inf.max()
else:
    max_num = dmd.max()
    
# input normalization: x -> [-1,1], because the last activation func in G is tanh
spd = min_max_normal(spd)
inf = min_max_normal(inf)
dmd = min_max_normal(dmd)
period = min_max_normal(period)

spd = torch.tensor(spd)
inf = torch.tensor(inf)
dmd = torch.tensor(dmd)
period = torch.tensor(period)

# prepare train set and test set
sz = int(int(spd.size(0)/12)*0.8) * 12
spd_train, spd_test = torch.split(spd,[sz, spd.size(0) - sz], dim = 0)
inf_train, inf_test = torch.split(inf,[sz, spd.size(0) - sz], dim = 0)
dmd_train, dmd_test = torch.split(dmd,[sz, spd.size(0) - sz], dim = 0)
period_train, period_test = torch.split(period,[sz, spd.size(0) - sz], dim = 0)

dataset = Data.TensorDataset(spd_train, inf_train, dmd_train, period_train)
test_dataset = Data.TensorDataset(spd_test, inf_test, dmd_test, period_test)

################################ CBML ##########################################
d_encoder = Encoder2(4, opt.D_out, opt.num_layers, opt.z_dim, domain=opt.domains) # Encode domain information
t_encoder = Encoder(4, opt.D_out, opt.num_layers, opt.z_dim, domain=False) # Encode task information
decoder = Decoder(3, opt.D_out, opt.z_dim)
vae = DVAE2(d_encoder, t_encoder, decoder)

if torch.cuda.device_count() > 1:
    print("number of GPU: ", torch.cuda.device_count())
    vae = nn.DataParallel(vae).to(device)
if torch.cuda.device_count() == 1:
    vae = vae.to(device)
for i in range(len(vae.d_encoder.domain_net)):
        vae.d_encoder.domain_net[i] = vae.d_encoder.domain_net[i].to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

def loss_function(y_test, y_pred, d_mu, d_log_var, t_mu, t_log_var):
    MSE = F.mse_loss(y_pred, y_test)
    KLD = -0.5 * torch.sum(1 + t_log_var - t_mu.pow(2) - t_log_var.exp()) -0.5 * torch.sum(1 + d_log_var - d_mu.pow(2) - d_log_var.exp())
    return MSE + KLD

################################ Training ##########################################
########################################################################################

def train(task_iter, train_loader, task_length):
    vae.train()
    train_loss = 0
    optimizer.zero_grad()
    for step, (speed, inflow, demand, sub_period) in enumerate(train_loader):
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
            # prepar domain data
            domain_data = prepare_domain(x_train[:t+1], y_train[:t+1], opt.domains, device)
            # prepary xy_train 
            x = x_train[t].view(1, -1, region_width, region_length)
            y = y_train[t].view(1, -1, region_width, region_length)
            xy_train = torch.cat((x, y), dim=1) # shape: (1, 4, 5, 5)
            y_true = torch.FloatTensor([y_test[t]])

            xy_train = Variable(xy_train.to(device))
            y_true = Variable(y_true.to(device))

            y_pred, (d_mu, d_log_var, d_h), (t_mu, t_log_var, t_h) = vae(xy_train, domain_data, d_h, t_h) #  d_h, t_h
            loss = loss_function(y_true * max_num, y_pred * max_num, d_mu, d_log_var, t_mu, t_log_var)   
            train_loss += loss

    train_loss.backward(retain_graph=True)
    optimizer.step()
    if opt.log_wandb:
            wandb.log({"Train Loss": train_loss / task_length / (step+1), 
                       "epoch": task_iter})
    print(f"Epoch: {task_iter} Train Loss: {train_loss / task_length / (step+1)}")

if __name__ == '__main__':
    for task_iter in range(opt.task_iteration):
        print("Start training iteration: ", task_iter)
        task_length = random.randint(12, 336) # random sample a length ranging from 1 day to 4 weeks can change to longer value
        Sampler = task_sampler(dataset, task_length)
        train_loader = Data.DataLoader(dataset=dataset, batch_size=task_length, shuffle=False, sampler=Sampler)

        train(task_iter, train_loader, task_length)
        
        if task_iter % 10 == 0:
            test_rmse, test_mape = evaluate(vae, test_dataset, opt, task_iter, max_num, device)
            if opt.log_wandb:
                wandb.log({"Average Test RMSE": test_rmse, 
                           "Average Test MAPE": test_mape,
                           "epoch": task_iter})
            print(f"Epoch {task_iter} Average TEST RMSE: ", test_rmse)
            print(f"Epoch {task_iter} Average TEST MAPE: ", test_mape)
        if task_iter % 50 == 0:
            save_model(vae, opt, task_iter, ckpt_dir)
    save_model(vae, opt, task_iter, ckpt_dir)
