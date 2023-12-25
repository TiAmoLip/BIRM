import argparse
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
from tqdm import tqdm
import pandas as pd
import wandb
import os
from torch import nn, optim, autograd
from model import EBD

from model import FeatureExtractor,AutoEncoder,Classifier
from torch.nn.functional import binary_cross_entropy_with_logits
from utils import eval_acc_class,mean_accuracy_class,pretty_print
from utils import CMNIST_LYDP,make_mnist_envs,concat_envs

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prior_sd_coef', type=int, default=1200)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=20)
parser.add_argument('--shape', type=int,default=28,help="shape of colored mnist",choices=[28,14])
parser.add_argument('--data_num', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="birm", type=str, choices=["birm", "irmv1", "erm"])

parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--step_gamma', type=float, default=0.1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=int, default=0)
parser.add_argument('--sampleN', type=int, default=10)
parser.add_argument('--wandb_log_freq',type=int,default=-1)
parser.add_argument('--model',type=str,default='MLP',choices=['MLP','CNN'])
parser.add_argument('--device',type=int,default=-1)



flags = parser.parse_args()
irm_type = flags.irm_type

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)
random.seed(1) # Fix the random seed of dataset
# Because random package is used to generate CifarMnist dataset
# We fix the randomness of the dataset.
if flags.device>=0:
    torch.set_default_device(f"cuda{flags.device}")
if flags.wandb_log_freq >0:
    wandb.login(key="433d80a0f2ec170d67780fc27cd9d54a5039a57b")
    wandb.init(project="BIRM",config=flags)
    
envs = make_mnist_envs(flags)
train_envs = envs[:-1]
combined_envs = concat_envs(train_envs,cuda=flags.device>=0)
test_envs = envs[-1:]
f_e = FeatureExtractor(flags)
q_u_e = [AutoEncoder(flags) for _ in train_envs]
q_u = AutoEncoder(flags)

# _lambda_ = 0.1

with tqdm(total=flags.steps) as pbar:
    _lambda_ = 0
    for step in range(flags.steps):
        f_e.train()
        # classifier.eval()
        for que, env_tr in zip(q_u_e,train_envs):
            que.fit(env_tr['images'],env_tr['labels'],f_e,20)
        q_u.fit(combined_envs[0],combined_envs[1],f_e,20)
        loss = 0
        opt = torch.optim.Adam(f_e.parameters(),lr=1e-3)
        for _ in range(10):
            epsilon = torch.randn_like(q_u.m_u)
            [ loss := loss + (1+_lambda_)*q_u.recon_loss(env_tr['images'],env_tr['labels'], q_u.sample(epsilon), f_e)-_lambda_*q.recon_loss(env_tr['images'],env_tr['labels'], q.sample(epsilon), f_e)  for q, env_tr in zip(q_u_e, train_envs) ]
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        
        epsilon = torch.zeros_like(q_u.m_u)
        classifier = q_u.sample(epsilon)
        if step>2000:
            _lambda_=0.1
            f_e.eval()
            classifier.eval()
            with torch.no_grad():
                test_logits = classifier(f_e(test_envs[0]['images']))
            test_acc,_,_ = eval_acc_class(test_logits,test_envs[0]['labels'],test_envs[0]['color'])
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item()},test_acc: {test_acc}")
        # if step%50==0:
        #     print(test_logits[:10])
        else:
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item()}")
if flags.wandb_log_freq>0:
    wandb.finish()