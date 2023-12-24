import argparse
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
import pandas as pd
import wandb
import os
from torch import nn, optim, autograd
from model import EBD

from model import MLP,CNN

from utils import eval_acc_class,mean_nll_class,mean_accuracy_class,pretty_print
from utils import CMNIST_LYDP



parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--prior_sd_coef', type=int, default=1200)
parser.add_argument('--dataset', type=str, default="CMNIST", choices=["CMNIST"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=20)
parser.add_argument('--shape', type=int,default=28,help="shape of colored mnist",choices=[28,14])
parser.add_argument('--data_num', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="birm", type=str, choices=["birm", "irmv1", "erm"])
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--step_gamma', type=float, default=0.1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=int, default=0)
parser.add_argument('--sampleN', type=int, default=20)
parser.add_argument('--wandb_log_freq',type=int,default=-1)
parser.add_argument('--model',type=str,default='MLP',choices=['MLP','CNN'])
flags = parser.parse_args()
irm_type = flags.irm_type

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)
random.seed(1) # Fix the random seed of dataset
# Because random package is used to generate CifarMnist dataset
# We fix the randomness of the dataset.
if flags.wandb_log_freq >0:
    wandb.login(key="433d80a0f2ec170d67780fc27cd9d54a5039a57b")
    wandb.init(project="BIRM",config=flags)

final_train_accs = []
final_test_accs = []
best_acc = 0
best_state_dict = []
for restart in range(flags.n_restarts):

    if flags.dataset == "CMNIST":
        dp = CMNIST_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        model = MLP(flags).cuda() if flags.model == "MLP" else CNN(flags).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
        flags.env_type = "linear"

    else:
        raise Exception
    if flags.opt == "adam":
        optimizer = optim.Adam(
          model.parameters(),
          lr=flags.lr)
    elif flags.opt == "sgd":
        optimizer = optim.SGD(
          model.parameters(),
          momentum=0.9,
          lr=flags.lr)
    else:
        raise Exception

    ebd = EBD(flags).cuda()
    lr_schd = lr_scheduler.StepLR(
        optimizer,
        step_size=int(flags.steps/2),
        gamma=flags.step_gamma)

    pretty_print('step', 'train loss', 'train penalty', 'test acc')

    for step in range(flags.steps):
        model.train()
        train_x, train_y, train_g, train_c= dp.fetch_train()

        sampleN = flags.sampleN
        train_penalty = 0
        train_logits = model(train_x)
        for i in range(sampleN):
            ebd.re_init_with_noise(flags.prior_sd_coef/flags.data_num)
            train_logits_w = ebd(train_g).view(-1, 1)*train_logits
            train_nll = mean_nll(train_logits_w, train_y)
            grad = autograd.grad(
                train_nll * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            train_penalty +=  1/sampleN * torch.mean(grad**2)
        train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c)
        weight_norm = torch.tensor(0.).cuda()
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
            if step >= flags.penalty_anneal_iters else 0.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= (1. + penalty_weight)
        penalty_weight*=1.002
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schd.step()


        
        if step % flags.print_every == 0:
            if flags.dataset != 'CifarMnist':
                model.eval()
            test_acc_list = []
            test_minacc_list = []
            test_majacc_list = []
            data_num = []
            for ii in range(test_batch_num):
                test_x, test_y, test_g, test_c= test_batch_fetcher()
                test_logits = model(test_x)
                test_acc_, test_minacc_, test_majacc_ = eval_acc(test_logits, test_y, test_c)
                test_acc_list.append(test_acc_ * test_x.shape[0])
                test_minacc_list.append(test_minacc_ * test_x.shape[0])
                test_majacc_list.append(test_majacc_ * test_x.shape[0])
                data_num.append(test_x.shape[0])
            total_data = torch.Tensor(data_num).sum()
            test_acc, test_minacc, test_majacc = torch.Tensor(test_acc_list).sum()/total_data, torch.Tensor(test_minacc_list).sum()/total_data, torch.Tensor(test_majacc_list).sum()/total_data
            if test_acc>best_acc:
                best_state_dict.clear()
                best_state_dict.append(model.state_dict())
                best_acc = test_acc
            if flags.wandb_log_freq<=0:
                pretty_print(
                    np.int32(step),
                    loss.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy(),
                )
            if flags.wandb_log_freq>0 and step%flags.wandb_log_freq==0:
                wandb.log({
                    "test_acc":test_acc,
                    "train_acc":train_acc,
                    "train_penalty":train_penalty,
                    "train_loss":loss
                })
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print(f'Final test acc: {np.mean(final_test_accs)}, best_acc:{best_acc}')
torch.save(best_state_dict[0],f"test{round(best_acc,5)}.pth")
wandb.finish()