import gym, torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from copy import deepcopy
from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import os
import time
import json
import math
from tqdm import tqdm


INPUT_CH = 67
FEATURE_CH = 512
MLP_CH = 1024
from Env import MAX_EDGE_NUM

class mlp_resblock(nn.Module):
    def __init__(self, in_ch, ch, out_ch=None, block_num=3, is_in=False):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = is_in
        self.is_out = out_ch
        
        if self.is_in:
            self.in_mlp = nn.Sequential(*[
                nn.Linear(in_ch, ch), 
                nn.LeakyReLU(0.1, inplace=True)])
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Linear(ch, ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(ch, ch)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=True)]))
        if self.is_out:
            self.out_mlp = nn.Sequential(*[
            nn.Linear(ch, ch), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(ch, out_ch)])
            
    def forward(self, x):
        if self.is_in:
            x = self.in_mlp(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_mlp(x)
        return x


class mlp_resblock_relu(nn.Module):
    def __init__(self, in_ch, ch, out_ch=None, block_num=3, is_in=False, is_relu=True):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = is_in
        self.is_out = out_ch
        self.is_relu = is_relu
        
        if self.is_in:
            self.in_mlp = nn.Sequential(*[
                nn.Linear(in_ch, ch), 
                nn.LeakyReLU(0.1, inplace=True)])
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Linear(ch, ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(ch, ch)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=True)]))
        if self.is_out:
            self.out_mlp = nn.Sequential(*[
            nn.Linear(ch, ch), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(ch, out_ch)
            ])
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        if self.is_in:
            x = self.in_mlp(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_mlp(x)
        if self.is_relu:
            x = self.relu(x)
        return x


class conv_resblock(nn.Module):
    def __init__(self, in_ch=None, ch=256, out_ch=None, block_num=3, is_relu=True):
        super().__init__()
        self.models=nn.Sequential()
        self.relus=nn.Sequential()
        self.block_num = block_num
        self.is_in = in_ch
        self.is_out = out_ch
        self.is_relu = is_relu
        
        if self.is_in:
            self.in_conv = nn.Sequential(*[
            nn.Conv1d(in_ch, ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=False)])
        for i in range(self.block_num):
            self.models.add_module(str(i), nn.Sequential(*[
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.1, inplace=False),
                nn.Conv1d(ch, ch, kernel_size=1, stride=1, padding=0)]))
            self.relus.add_module(str(i), nn.Sequential(*[
                nn.LeakyReLU(0.1, inplace=False)]))
        if self.is_out:
            self.out_conv = nn.Conv1d(ch, out_ch, kernel_size=1, stride=1, padding=0)
        if self.is_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.is_in:
            x = self.in_conv(x)
        for i in range(self.block_num):
            x0 = x
            x = self.models[i](x)
            x += x0
            x = self.relus[i](x)
        if self.is_out:
            x = self.out_conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x



class conv_mlp_net(nn.Module):
    def __init__(self, conv_in, conv_ch, mlp_in, mlp_ch, block_num=3, is_gpu=True):
        super().__init__()
        input_dim = 36
        action_dim = 2
        self.is_gpu = is_gpu
        self.mlp_in = mlp_in

        self.feature_network_A = conv_resblock(in_ch=conv_in, ch=conv_ch, out_ch=conv_ch, block_num=block_num)
        self.feature_network_B = mlp_resblock(in_ch=mlp_in, ch=mlp_ch, out_ch=mlp_ch, block_num=3, is_in=True)

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs):

        x = self.feature_network_A(obs)
        x = x.view(-1, self.mlp_in)
        x = self.feature_network_B(x)
        
        return x



class feature_transfer_net(nn.Module):
    def __init__(self, mode='actor', is_gpu=True):
        super().__init__()
        self.is_gpu = is_gpu
        self.mode = mode
        
        self.feature_network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(MAX_EDGE_NUM+1)*FEATURE_CH, mlp_ch=MLP_CH, block_num=3)
        
        self.bais_network1 = mlp_resblock(in_ch=2, ch=MLP_CH, block_num=3, is_in=True)
        # self.bais_network2 = mlp_resblock(in_ch=MAX_EDGE_NUM+1, ch=MLP_CH, block_num=3, is_in=True)
        
        if self.mode == 'actor':
            self.transfer_network = mlp_resblock(in_ch=MLP_CH, ch=MLP_CH, out_ch=(MAX_EDGE_NUM+1), block_num=3, is_in=False)
        else:
            self.transfer_network = mlp_resblock(in_ch=MLP_CH, ch=MLP_CH, out_ch=(MAX_EDGE_NUM+1), block_num=3, is_in=False)

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
        state = obs['servers']
        preference = obs['preference']
        freq = obs['frequency']
        mask1 = obs['mask1']
        mask2 = obs['mask2']
        
        state = torch.tensor(state).float()
        preference = torch.tensor(preference).float()
        freq = torch.tensor(freq).float()
        mask1 = torch.tensor(mask1).float()
        mask2 = torch.tensor(mask2).float()
        if self.is_gpu:
            state = state.cuda()
            preference = preference.cuda()
            freq = freq.cuda()
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()

        f = self.feature_network(state)
        b_w = self.bais_network1(preference)
        # b_f = self.bais_network2(freq)
        x = f + b_w #+ b_f
        logits = self.transfer_network(x)
        # logits = logits.view(-1, MAX_EDGE_NUM+1)
        mask = mask2==0
        # if self.mode == 'actor':
        logits.masked_fill_(mask, -1e6)

        return logits, state


class Actor(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()
        
        self.is_gpu = is_gpu

        self.net = feature_transfer_net(mode='actor')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        logits,_ = self.net(obs)

        return logits, state



class Critic(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()

        self.is_gpu = is_gpu

        self.net = feature_transfer_net(mode='critic')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        v,_ = self.net(obs)

        return v
    