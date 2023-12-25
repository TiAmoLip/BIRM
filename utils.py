
import math
import numpy as np
import torch
from torchvision import datasets


from torch import nn, optim, autograd


def update_flags(flags):
    flags.prior_sd_coef = 1200
    return flags

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def torch_xor(a, b):
    return (a-b).abs()

def concat_envs(con_envs,cuda = False):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat([
        ig * torch.ones_like(env["labels"])
        for ig,env in enumerate(con_envs)])
    # con_2g = torch.cat([
    #     (ig < (len(con_envs) // 2)) * torch.ones_like(env["labels"])
    #     for ig,env in enumerate(con_envs)]).long()
    con_c = torch.cat([env["color"] for env in con_envs])
    # con_yn = torch.cat([env["noise"] for env in con_envs])
    if cuda:
        return con_x.cuda(), con_y.cuda(), con_g.cuda(), con_c.cuda()
    else:
        return con_x, con_y, con_g, con_c


def eval_acc_class(logits, labels, colors):
    acc  = mean_accuracy_class(logits, labels)
    minacc = mean_accuracy_class(
      logits[colors!=1],
      labels[colors!=1])
    majacc = mean_accuracy_class(
      logits[colors==1],
      labels[colors==1])
    return acc, minacc, majacc

def mean_accuracy_class(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()




def make_environment(images, labels, e, shape=28,cuda=False):
    # 2x subsample for computational convenience
    if shape==14:
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    else:
        images = images.reshape((-1, 28, 28))
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)).cuda() if cuda else torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    color_mask = torch_bernoulli(e, len(labels)).cuda() if cuda else torch_bernoulli(e, len(labels)) 
    colors = torch_xor(labels, color_mask)
    # colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'color': (1- color_mask[:, None])
    }


def make_mnist_envs(flags):
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:flags.data_num], mnist.targets[:flags.data_num])
    mnist_val = (mnist.data[flags.data_num:], mnist.targets[flags.data_num:])
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    # Build environments
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        for i in range(envs_num):
            envs.append(
              make_environment(
                  mnist_train[0][i::envs_num],
                  mnist_train[1][i::envs_num],
                  (0.2 - 0.1)/(envs_num-1) * i + 0.1,flags.shape,flags.device>=0))
    elif flags.env_type == "sin":
        for i in range(envs_num):
            envs.append(
                make_environment(mnist_train[0][i::envs_num], mnist_train[1][i::envs_num], (0.2 - 0.1) * math.sin(i * 2.0 * math.pi / (envs_num-1)) * i + 0.1,flags.shape,flags.device>=0))
    elif flags.env_type == "step":
        lower_coef = 0.1
        upper_coef = 0.2
        env_per_group = flags.envs_num // 2
        for i in range(envs_num):
            env_coef = lower_coef if i < env_per_group else upper_coef
            envs.append(
                make_environment(
                    mnist_train[0][i::envs_num],
                    mnist_train[1][i::envs_num],
                    env_coef,
                    flags.shape,
                    flags.device>=0
                    ))
    else:
        raise Exception
    envs.append(make_environment(mnist_val[0], mnist_val[1], 0.9,flags.shape,flags.device>=0))
    return envs


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass

class IRMDataProvider(LYDataProvider):
    def __init__(self, flags):
        super(IRMDataProvider, self).__init__()

    def preprocess_data(self):
        self.train_x, self.train_y, self.train_g, self.train_c= concat_envs(self.envs[:-1])
        self.test_x, self.test_y, self.test_g, self.test_c= concat_envs(self.envs[-1:])

    def fetch_train(self):
        return self.train_x, self.train_y, self.train_g, self.train_c

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_g, self.test_c

class CMNIST_LYDP(IRMDataProvider):
    def __init__(self, flags):
        super(CMNIST_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_mnist_envs(flags)
        self.preprocess_data()



