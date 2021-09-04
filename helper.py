import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import random
from pathlib import Path

# Normalization: Necessary for Tanh
def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    return x

city_width = 50
city_length = 50

region_width = 5
region_length = 5

# Process data
def preprocess_data(opt, path='~/CBML/data/'):
    speed = np.loadtxt(open(path+'speed_city.csv', "rb"), delimiter=",", skiprows=0)
    inflow = np.loadtxt(open(path+'inflow_city.csv', "rb"), delimiter=",", skiprows=0)
    demand = np.loadtxt(open(path+'demand_city.csv', "rb"), delimiter=",", skiprows=0)

    speed = speed.reshape(-1, city_width, city_length)
    inflow = inflow.reshape(-1, city_width, city_length)
    demand = demand.reshape(-1, city_width, city_length)

    spd = speed[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
    inf = inflow[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
    dmd = demand[:, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]

    num_day = int(spd.shape[0] / 12)
    period_per_day = np.arange(12)
    period = np.tile(period_per_day, int(speed.shape[0] / 12))  # 1-D vector
    return (spd, inf, dmd), num_day, period_per_day, period

# Sampling tasks
class task_sampler(Data.Sampler):
    r"""Samples tasks of random length.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, task_length):
        self.data_source = data_source
        self.task_length = task_length

    def __iter__(self):
        n = len(self.data_source)
        all_indices = torch.arange(0, n, dtype=torch.int64).unfold(0, self.task_length, self.task_length)   # size tenser(m, task_length)
        return iter(all_indices.contiguous().view(-1,).tolist())    # convert tenser(m, task_length) to i-d tensor and to list

    def __len__(self):
        return len(self.data_source)

# Save model
def save_model(vae, opt, task_iter, ckpt_dir):
    torch.save(vae.state_dict(), ckpt_dir / f"VAE_params_{opt.task_iteration}_region({opt.region_i},{opt.region_j})_ +{opt.D_out}_{opt.z_dim}_{opt.num_layers}_Epoch_{task_iter}.pkl")
    
def target_identifier(target, speed, inflow, demand):
    if target == "speed":
        return speed, inflow, demand
    elif target == "inflow":
        return inflow, speed, demand
    else:
        return demand, inflow, speed
    
def prepare_domain(x, y, domains, device):
    length = x.size()[0]
    xy_train = torch.cat((x, y), dim=1) 
    domain_data = []
    if "COMBO" in domains: 
        domain_data.append(Variable(torch.mean(xy_train[(length-1-(length-1)//12*12)::12, :, :, :], dim=0).view(1, -1, region_width, region_length)).to(device))
        return domain_data
    for domain in domains:
        if domain == "daily": 
            p = 12
        elif domain == "weekly":
            p = 12*7
        elif domain == "monthly":
            p = 12*7*4
        begin_idx = length-1-(length-1)//p*p
        domain_data.append(Variable(torch.mean(xy_train[begin_idx::p, :, :, :], dim=0).view(1, -1, region_width, region_length)).to(device))
    return domain_data