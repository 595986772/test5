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
from Env import MEC_Env
from Model import Actor, Critic



config = 'multi-part'
actor_lr, critic_lr = 1e-6, 1e-6
epoch, batch_size = 1000, 1024*4
train_num, test_num = 64, 1024
gamma, lr_decay = 0.95, None
buffer_size = 1000000
buffer_alpha, buffer_beta = 0.6, 0.4
eps_train, eps_test = 0.1, 0.00
step_per_epoch, step_per_collect = 100*train_num*5, train_num*20
writer = SummaryWriter('log/sac')  # tensorboard is also supported!
logger = ts.utils.BasicLogger(writer)
is_gpu = True
#sac
tau, alpha = 0.005, 0.05
n_step = 1
rew_norm = False
recompute_adv = 0


cuda_device_num = 0
torch.cuda.set_device(cuda_device_num)
device = torch.device('cuda', cuda_device_num)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

##########################################################################################
# main

def main():
    actor = Actor()
    critic1 = Critic()
    critic2 = Critic()
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    load_path = None
    # load_path = 'save-sac/pth/exp1/ep10-actor.pth'
    # actor.load_model(load_path)
    # load_path = 'save-saac/pth/exp1/ep10-critic1.pth'
    # critic1.load_model(load_path)
    # load_path = 'save-sac/pth/exp1/ep10-critic2.pth'
    # critic2.load_model(load_path)

    if is_gpu:
        actor.cuda()
        critic1.cuda()
        critic2.cuda()


    policy = ts.policy.DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau,
        gamma,
        alpha,
        estimation_step=n_step,
        reward_normalization=rew_norm
    )

    
    def save_best_fn (policy):
        policy.actor.save_model('save-edge/pth/best/actor.pth')
        policy.critic1.save_model('save-edge/pth/best/critic1.pth')
        policy.critic2.save_model('save-edge/pth/best/critic2.pth')
        # pass

    def test_fn(epoch, env_step):
        policy.actor.save_model('save-edge/pth/exp1/ep%03d-actor.pth'%(epoch))
        policy.critic1.save_model('save-edge/pth/exp1/ep%03d-critic1.pth'%(epoch))
        policy.critic2.save_model('save-edge/pth/exp1/ep%03d-critic2.pth'%(epoch))
        # pass

    def train_fn(epoch, env_step):
        pass
        # policy.set_eps(eps_train)

    train_envs = DummyVectorEnv([lambda i=i: MEC_Env(conf_name=config, w=i/(train_num-1)) for i in range(train_num)])
    test_envs = DummyVectorEnv([lambda i=i: MEC_Env(conf_name=config, w=i/(test_num-1)) for i in range(test_num)])

    buffer = ts.data.VectorReplayBuffer(buffer_size, train_num)
    train_collector = ts.data.Collector(policy, train_envs, buffer)
    test_collector = ts.data.Collector(policy, test_envs)  # because DQN uses epsilon-greedy method
    train_collector.collect(n_episode=train_num)

    result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
            test_num, batch_size, update_per_step=10 / step_per_collect,
            train_fn = train_fn,
            test_fn=test_fn,
            stop_fn=None,
            save_best_fn = save_best_fn,
            logger=logger)



##########################################################################################

if __name__ == "__main__":
    main()
