import argparse
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
from tqdm import tqdm
import wandb

from model import EBD

from model import FeatureExtractor,AutoEncoder,Classifier
# from torch.nn.functional import binary_cross_entropy_with_logits
from utils import eval_acc_class,mean_accuracy_class,pretty_print
from utils import CMNIST_LYDP,make_mnist_envs,concat_envs

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--shape', type=int,default=28,help="shape of colored mnist",choices=[28,14])
parser.add_argument('--data_num', type=int,default=20000,help="shape of colored mnist")
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])

parser.add_argument('--hidden_dim', type=int, default=16)

parser.add_argument('--penalty_anneal_iters', type=int, default=10)
parser.add_argument('--_lambda_', type=float, default=1.0)
parser.add_argument('--steps', type=int, default=501)

parser.add_argument('--sampleN', type=int, default=10)
parser.add_argument('--wandb_log_freq',type=int,default=-1)
parser.add_argument('--device',type=int,default=-1)



flags = parser.parse_args()

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)
random.seed(1) # Fix the random seed of dataset
# Because random package is used to generate CifarMnist dataset
# We fix the randomness of the dataset.
if flags.device>=0:
    torch.set_default_device(f"cuda:{flags.device}")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
if flags.wandb_log_freq >0:
    wandb.login(key="433d80a0f2ec170d67780fc27cd9d54a5039a57b")
    t = random.randint(0,1000)
    q = random.randint(2000,100000)
    wandb.init(project="BIRM",config=flags,name=f"bayesian_prob-{t}_{q}")
    
envs = make_mnist_envs(flags)
train_envs = envs[:-1]
combined_envs = concat_envs(train_envs,cuda=flags.device>=0)
test_envs = envs[-1:]
f_e = FeatureExtractor(flags)
q_u_e = [AutoEncoder(flags) for _ in train_envs]
q_u = AutoEncoder(flags)

# _lambda_ = 0.1
# q_u.m_u = 1
# for que, env_tr in zip(q_u_e,train_envs):
#     que.m_u = 1


opt = torch.optim.Adam(f_e.parameters(),lr=flags.lr)
with tqdm(total=flags.steps) as pbar:
    _lambda_ = 0
    for step in range(flags.steps):
        opt.zero_grad()
        f_e.train()
        # classifier.eval()
        for que, env_tr in zip(q_u_e,train_envs):
            # que.std = torch.normal(1,0.5)
            que.reinit_std()
            # que.fit(env_tr['images'],env_tr['labels'],f_e,10)
        # q_u.fit(combined_envs[0],combined_envs[1],f_e,10)
        q_u.reinit_std()
        loss = 0
        for _ in range(flags.sampleN):
            epsilon = torch.randn_like(q_u.m_u)
            [ loss := loss + q_u.nll(env_tr['images'],env_tr['labels'], q_u.sample(epsilon), f_e) + _lambda_*(q_u.nll(env_tr['images'],env_tr['labels'], q_u.sample(epsilon), f_e)-q.nll(env_tr['images'],env_tr['labels'], q.sample(epsilon), f_e))  for q, env_tr in zip(q_u_e, train_envs) ]
        weight_norm = torch.tensor(0.)
        for w in f_e.parameters():
            weight_norm += w.norm().pow(2)
        
        loss = loss/flags.sampleN
        loss += flags.l2_regularizer_weight*weight_norm
        if _lambda_ > 1.0:
            loss /= (1. + _lambda_)
        loss.backward()
        opt.step()
        
        
        epsilon = torch.zeros_like(q_u.m_u)
        classifier = q_u.sample(epsilon)
        if step>flags.penalty_anneal_iters:
            _lambda_=flags._lambda_
            f_e.eval()
            classifier.eval()
            with torch.no_grad():
                test_logits = classifier(f_e(test_envs[0]['images']))
                train_logits = classifier(f_e(combined_envs[0]))
            test_acc,_,_ = eval_acc_class(test_logits,test_envs[0]['labels'],test_envs[0]['color'])
            train_acc,_,_ = eval_acc_class(train_logits,combined_envs[1],combined_envs[-1])
            
            pbar.update(1)

            pbar.set_description(f"train_loss: {round(loss.item(),5)}, train_acc: {round(train_acc.item(),5)}:,test_acc: {round(test_acc.item(),5)}")
            if step%flags.wandb_log_freq==0 and flags.wandb_log_freq>0:
                wandb.log({
                    "train_loss":loss.item(),
                    "test_acc":test_acc.item(),
                    "train_acc":train_acc.item()
                })

        else:
            f_e.eval()
            classifier.eval()
            with torch.no_grad():
                # test_logits = classifier(f_e(test_envs[0]['images']))
                train_logits = classifier(f_e(combined_envs[0]))
            # test_acc,_,_ = eval_acc_class(test_logits,test_envs[0]['labels'],test_envs[0]['color'])
            train_acc,_,_ = eval_acc_class(train_logits,combined_envs[1],combined_envs[-1])
            pbar.update(1)
            pbar.set_description(f"train_loss: {round(loss.item(),5)}, train_acc:{train_acc}")
if flags.wandb_log_freq>0:
    wandb.finish()
# python train.py --l2_regularizer_weight 0.004 --hidden_dim 3 --_lambda_ 100 --steps 1000 --wandb_log_freq 20 --lr 0.0004看起来能收敛