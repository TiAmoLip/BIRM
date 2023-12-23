from torch import nn, optim, autograd
import pdb
import torch
from torchvision import datasets

class ENV_EBD(nn.Module):
    def __init__(self, flags):
      super(ENV_EBD, self).__init__()
      self.embedings = torch.nn.Embedding(flags.envs_num, 4)
      self.re_init()

    def re_init(self):
      pass
      # self.embedings.weight.data.fill_(1.)

    def forward(self, e):
      return self.embedings(e.long())

class EBD(nn.Module):
    def __init__(self, flags):
      super(EBD, self).__init__()
      self.flags = flags
      # if self.flags.num_classes == 2:
      self.embedings = torch.nn.Embedding(flags.envs_num, 1)
      # else:
      #     self.embedings = torch.nn.Embedding(flags.envs_num, self.flags.num_classes)
      self.re_init()

    def re_init(self):
      self.embedings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd):
      # if self.flags.num_classes == 2:
      rd = torch.normal(
         torch.Tensor([1.0] * self.flags.envs_num),
         torch.Tensor([noise_sd] * self.flags.envs_num))
      self.embedings.weight.data = rd.view(-1, 1).cuda()
      # else:
      # rd = torch.normal(
      #    torch.Tensor([1.0] * self.flags.envs_num * self.flags.num_classes),
      #    torch.Tensor([noise_sd] * self.flags.envs_num* self.flags.num_classes))
      # self.embedings.weight.data = rd.view(-1, self.flags.num_classes).cuda()

    def forward(self, e):
      return self.embedings(e.long())


class Y_EBD(nn.Module):
    def __init__(self, flags):
      super(Y_EBD, self).__init__()
      self.embedings = torch.nn.Embedding(flags.classes_num, 4)
      self.re_init()

    def re_init(self):
      pass
      # self.embedings.weight.data.fill_(1.)

    def forward(self, e):
      return self.embedings(e.long())


class BayesW(nn.Module):
    def __init__(self, prior, flags, update_w=True):
        super(BayesW, self).__init__()
        self.pw, self.psigma = prior
        self.flags = flags
        self.vw = torch.nn.Parameter(self.pw.clone(), requires_grad=update_w)
        self.vsigma= torch.nn.Parameter(self.psigma.clone())
        self.nll = nn.MSELoss()
        self.re_init()

    def reset_prior(self, prior):
        self.pw, self.psigma = prior
        print("resetting prior", self.pw.item(), self.psigma.item())

    def reset_posterior(self, prior):
        new_w, new_sigma = prior
        self.vw.data, self.vsigma.data = new_w.clone(), new_sigma.clone()
        print("resetting posterior", self.pw.item(), self.psigma.item())


    def generate_rand(self, N):
        self.epsilon = list()
        for i in range(N):
            self.epsilon.append(
                torch.normal(
                    torch.tensor(0.0),
                    torch.tensor(1.0)))

    def variational_loss(self, xb, yb, N):
        pw, psigma = self.pw, self.psigma
        vw, vsigma = self.vw, self.vsigma
        kl = torch.log(psigma/vsigma) + (vsigma ** 2 + (vw - pw) ** 2) / (2 * psigma ** 2)
        lk_loss = 0
        assert N == len(self.epsilon)
        for i in range(N):
            epsilon_i = self.epsilon[i]
            wt_ei = vw + vsigma * epsilon_i
            loss_i = self.nll(wt_ei * xb, yb)
            lk_loss += 1./N * loss_i
        return lk_loss + 1./self.flags.data_num  * kl

    def forward(self, x):
        return self.vw * x

    def re_init(self):
      pass

    def init_sep_by_share(self, share_bayes_net):
        self.vw.data = share_bayes_net.vw.data.clone()
        self.vsigma.data = share_bayes_net.vsigma.data.clone()
        self.epsilon = share_bayes_net.epsilon


# class MLP(nn.Module):
#     def __init__(self, flags):
#         super(MLP, self).__init__()
#         self.flags = flags
#         if flags.grayscale_model:
#           lin1 = nn.Linear(14 * 14, flags.hidden_dim)
#         else:
#           lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
#         lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
#         lin3 = nn.Linear(flags.hidden_dim, 1)
#         for lin in [lin1, lin2, lin3]:
#           nn.init.xavier_uniform_(lin.weight)
#           nn.init.zeros_(lin.bias)
#         self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

#     def forward(self, input):
#         if self.flags.grayscale_model:
#           out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
#         else:
#           out = input.view(input.shape[0], 2 * 14 * 14)
#         out = self._main(out)
#         return out
class MLP(nn.Module):
    def __init__(self, flags) -> None:

        super(MLP, self).__init__()
        self.flags = flags
        self.conv = nn.Sequential(
            nn.Conv2d(2,16,3,1,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7,100),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(100,1)
        )
    def forward(self, input):
        # if self.flags.grayscale_model:
        #     out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        # else:
        #     out = input.view(input.shape[0], 2 * 14 * 14)
        out = self.conv(out)
        return out


class MLPFull(nn.Module):
    def __init__(self, flags):
        super(MLPFull, self).__init__()
        self.flags = flags
        if flags.grayscale_model:
          lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
          lin1 = nn.Linear(3 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, flags.num_classes)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if self.flags.grayscale_model:
          out = input.view(input.shape[0], 3, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 3 * 14 * 14)
        out = self._main(out)
        return out


class PredEnvHatY(nn.Module):
    def __init__(self, flags):
        super(PredEnvHatY, self).__init__()
        self.lin1 = nn.Linear(1, flags.hidden_dim)
        self.lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2)

    def forward(self, input):
        out = self._main(input)
        return out


class InferEnv(nn.Module):
    def __init__(self, flags):
        super(InferEnv, self).__init__()
        self.lin1 = nn.Linear(1, flags.hidden_dim)
        self.lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.Sigmoid())

    def forward(self, input):
        out = self._main(input)
        return out



