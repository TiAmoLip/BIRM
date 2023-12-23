import argparse
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
import pandas as pd
# import wandb
import os
import sys
from torch import nn, optim, autograd
from model import EBD
# from model import resnet18_sepfc_us
from model import MLP
# torch.cuda.set_device("cpu")
sys.path.append('dataset_scripts')
from utils import concat_envs,eval_acc_class,eval_acc_reg,mean_nll_class,mean_accuracy_class,mean_nll_reg,mean_accuracy_reg,pretty_print, return_model
from utils import CMNIST_LYDP
# from utils import CIFAR_LYPD, COCOcolor_LYPD
# from utils import mean_nll_multi_class,eval_acc_multi_class,mean_accuracy_multi_class


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default="CMNIST", choices=["CifarMnist","ColoredObject", "CMNIST"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.01)
parser.add_argument('--print_every', type=int,default=2)
parser.add_argument('--data_num', type=int, default=20000)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="birm", type=str, choices=["birm", "irmv1", "erm"])
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--step_gamma', type=float, default=0.1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=100000.0)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--grayscale_model', type=int, default=0)
parser.add_argument('--device', type=int, default=-1, help="-1 means cpu, else cuda")
# parser.add_argument('--batch_size', type=int, default=300)

flags = parser.parse_args()
irm_type = flags.irm_type
if flags.device>=0:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.set_device(f"cuda:{flags.device}")

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)
random.seed(1) # Fix the random seed of dataset
# Because random package is used to generate CifarMnist dataset
# We fix the randomness of the dataset.


final_train_accs = []
final_test_accs = []
flags, model_type = return_model(flags)
for restart in range(flags.n_restarts):

    if flags.dataset == "CMNIST":
        dp = CMNIST_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = MLP(flags)
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
        flags.env_type = "linear"
    else:
        raise Exception
    if flags.opt == "adam":
        optimizer = optim.Adam(
          mlp.parameters(),
          lr=flags.lr)
    elif flags.opt == "sgd":
        optimizer = optim.SGD(
          mlp.parameters(),
          momentum=0.9,
          lr=flags.lr)
    else:
        raise Exception

    ebd = EBD(flags)
    lr_schd = lr_scheduler.StepLR(
        optimizer,
        step_size=int(flags.steps/2),
        gamma=flags.step_gamma)

    pretty_print('step', 'train loss', 'train penalty', 'test acc')
    # if flags.irm_type == "cirm_sep":
    #     pred_env_haty_sep.init_sep_by_share(pred_env_haty)
    # train_loader = dp.fetch_train()
    # test_loader = dp.fetch_test()
    for step in range(flags.steps):
        mlp.train()
        train_x, train_y, train_g, train_c= dp.fetch_train()
        
        
        sampleN = 10
        train_penalty = 0
        train_logits = mlp(train_x)
        for i in range(sampleN):
            ebd.re_init_with_noise(flags.prior_sd_coef/flags.data_num)
            train_logits_w = ebd(train_g).view(-1, 1)*train_logits
            train_nll = mean_nll(train_logits_w, train_y)
            grad = autograd.grad(
                train_nll * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            train_penalty +=  1/sampleN * torch.mean(grad**2)
        # elif irm_type == "erm":
        #     train_logits = mlp(train_x)
        #     train_nll = mean_nll(train_logits, train_y)
        #     train_penalty = torch.tensor(0.0)
        # else:
        #     raise Exception
        train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c)
        weight_norm = torch.tensor(0.)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
            if step >= flags.penalty_anneal_iters else 0.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= (1. + penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schd.step()

        if step % flags.print_every == 0:
            # if flags.dataset != 'CifarMnist':
            mlp.eval()
            test_acc_list = []
            test_minacc_list = []
            test_majacc_list = []
            data_num = []
            # for ii in range(test_batch_num):
            test_x, test_y, test_g, test_c= test_batch_fetcher()
            # for test_x, test_y, test_g, test_c in test_loader:
            with torch.no_grad():
                test_logits = mlp(test_x)
            test_acc_, test_minacc_, test_majacc_ = eval_acc(test_logits, test_y, test_c)
            test_acc_list.append(test_acc_ * test_x.shape[0])
            test_minacc_list.append(test_minacc_ * test_x.shape[0])
            test_majacc_list.append(test_majacc_ * test_x.shape[0])
            data_num.append(test_x.shape[0])
            total_data = torch.Tensor(data_num).sum()
            test_acc, test_minacc, test_majacc = torch.Tensor(test_acc_list).sum()/total_data, torch.Tensor(test_minacc_list).sum()/total_data, torch.Tensor(test_majacc_list).sum()/total_data
            pretty_print(
                np.int32(step),
                loss.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
            )
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final test acc: %s' % np.mean(final_test_accs))