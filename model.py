from torch import nn, optim, autograd
import pdb
import torch
from torch.nn import functional as F
from torchvision import datasets


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

    def re_init_with_noise(self,mean_sd=1, noise_sd=1200/20000):
      # if self.flags.num_classes == 2:
      rd = torch.normal(
         torch.Tensor([mean_sd] * self.flags.envs_num),
         torch.Tensor([noise_sd] * self.flags.envs_num))
      self.embedings.weight.data = rd.view(-1, 1).cuda()
      # else:
      # rd = torch.normal(
      #    torch.Tensor([1.0] * self.flags.envs_num * self.flags.num_classes),
      #    torch.Tensor([noise_sd] * self.flags.envs_num* self.flags.num_classes))
      # self.embedings.weight.data = rd.view(-1, self.flags.num_classes).cuda()

    def forward(self, e):
      return self.embedings(e.long())

class MLP(nn.Module):
    def __init__(self, flags):
        super(MLP, self).__init__()
        self.flags = flags
        self.shape = flags.shape
        if flags.grayscale_model:
          lin1 = nn.Linear(self.shape * self.shape, flags.hidden_dim)
        else:
          lin1 = nn.Linear(2 * self.shape * self.shape, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.LeakyReLU(0.1,True),nn.Dropout(0.2), lin2, nn.LeakyReLU(0.1,True),nn.Dropout(0.2), lin3)

    def forward(self, input):
        if self.flags.grayscale_model:
          out = input.view(input.shape[0], 2, self.shape * self.shape).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * self.shape * self.shape)
        out = self._main(out)
        return out
class CNN(nn.Module):
    def __init__(self, flags) -> None:

        super(CNN, self).__init__()
        self.shape = flags.shape
        self.flags = flags
        self.conv = nn.Sequential(
            nn.Conv2d(2,16,3,1,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(16,32,3,1,1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.MaxPool2d(2) if self.shape==28 else nn.Identity(),
            
            nn.Flatten(),
            nn.Linear(32*7*7,100),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(100,1)
        )
    def forward(self, input):
        # if self.flags.grayscale_model:
        #     out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        # else:
        #     out = input.view(input.shape[0], 2 * 14 * 14)
        out = self.conv(input.view(-1,2,self.shape,self.shape))
        return out



class FeatureExtractor(nn.Module):

    def __init__(self,flags) -> None:
        super(FeatureExtractor, self).__init__()
        # self.lin = nn.Linear(4,2,bias=False)
        self.flags = flags
        self.l = nn.Sequential(
            nn.Linear(2*flags.shape*flags.shape,flags.hidden_dim),
            nn.LeakyReLU(0.2,True),
            nn.Linear(flags.hidden_dim,flags.hidden_dim)
          )

    def forward(self, x):
        return self.l(x.view(-1,2*self.flags.shape*self.flags.shape))

class Classifier(nn.Module):

    def __init__(self, mean, std, epsilon) -> None:
        super(Classifier, self).__init__()
        self.w = mean + std*epsilon
    
    def forward(self, x):
        return torch.matmul(x, self.w.T)

class AutoEncoder(nn.Module):

    def __init__(self,flags) -> None:
        super(AutoEncoder, self).__init__()
        self.m_u = torch.nn.parameter.Parameter(torch.rand(1,flags.hidden_dim))
        self.std = torch.nn.parameter.Parameter(torch.rand(1,flags.hidden_dim))
        torch.nn.init.uniform_(self.m_u, -1, 1)
        torch.nn.init.uniform_(self.std, 0, 1)
    
    def recon_loss(self, X, Y, classifier, f_e): #minimize this
        return F.binary_cross_entropy_with_logits(classifier(f_e(X)),Y)
    
    def KL_loss(self):   #maximize this
        return torch.sum(1 + torch.log(torch.square(self.std)) - torch.square(self.m_u) - torch.square(self.std))/2

    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon =  torch.randn_like(self.m_u)
        return Classifier(self.m_u, self.std, epsilon)
    
    def fit(self, X, Y, f_e, epochs):
        optim = torch.optim.Adam([self.m_u, self.std], betas=(0.5, 0.5))
        for _ in range(epochs):
            loss = self.recon_loss(X, Y, self.sample(), f_e) - self.KL_loss()
            optim.zero_grad()
            loss.backward()
            optim.step()




