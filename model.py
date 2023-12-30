from torch import nn, optim, autograd
import pdb
import torch
import numpy as np
from torch.nn import functional as F
from torchvision import datasets
# from scipy.integrate import quad

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

    def reinit_std(self,mean_sd=1, noise_sd=1200/20000):
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

        lin1 = nn.Linear(2 * self.shape * self.shape, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.LeakyReLU(0.1,True),nn.Dropout(0.2), lin2, nn.LeakyReLU(0.1,True),nn.Dropout(0.2), lin3)

    def forward(self, input):
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

        out = self.conv(input.view(-1,2,self.shape,self.shape))
        return out



class FeatureExtractor(nn.Module):

    def __init__(self,flags) -> None:
        super(FeatureExtractor, self).__init__()
        # self.lin = nn.Linear(4,2,bias=False)
        self.flags = flags
        self.l = nn.Sequential(
            nn.Linear(2*flags.shape*flags.shape,flags.hidden_dim),
            # nn.LeakyReLU(0.2,True),
            # nn.Linear(flags.shape,flags.shape),
            # nn.LeakyReLU(0.2,True),
            # nn.Linear(flags.shape,flags.hidden_dim)
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
        # self.m_u = torch.nn.parameter.Parameter(torch.rand(1,flags.hidden_dim))
        # self.std = torch.nn.parameter.Parameter(torch.rand(1,flags.hidden_dim))
        self.m_u = torch.Tensor([[1]*flags.hidden_dim])
        self.std = torch.Tensor([[0.5]*flags.hidden_dim])
        # torch.nn.init.uniform_(self.m_u, -1, 1)
        # torch.nn.init.uniform_(self.std, 0, 1)
        self.s = 0
    
    def nll(self, X, Y, classifier, f_e): #minimize this
        # s = torch.randn(1)*self.m_u + self.std
        return F.binary_cross_entropy_with_logits(classifier(f_e(X)*self.s),Y)
    def reinit_s(self):
      self.s = torch.randn(1)*self.m_u.detach() + self.std.detach()
    def reinit_std(self,noise=0.5):
      # print(self.m_u.shape)
      self.std = torch.randn(*self.std.shape)*noise+1
    
    def KL_Div(self):   #maximize this
      # mu = self.m_u.detach().numpy()
      # sigma = self.std.detach().numpy()
      # p = lambda x: 1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/2/sigma**2)
      # # q = lambda x: 1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/2/sigma**2)
      # integrand = lambda x: p(x)*(-np.log(sigma)+0.5-(x-mu)**2/2/sigma**2)
      
      # integrand = lambda x: p(x)*(-np.log(sigma)+np.log(np.exp(0.5-(x-mu)**2/2/sigma**2)))
      # result, _ = quad(integrand,-100,100)
      # return result
      
      return torch.sum(1 + torch.log(torch.square(self.std)) - torch.square(self.m_u) - torch.square(self.std))/2

    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon =  torch.randn_like(self.m_u)
        return Classifier(self.m_u, self.std, epsilon)
    
    def fit(self, X, Y, f_e, epochs):
        optim = torch.optim.Adam([self.m_u, self.std],lr=1e-3)
        for _ in range(epochs):
            loss = self.nll(X, Y, self.sample(), f_e) - self.KL_Div()
            optim.zero_grad()
            loss.backward()
            optim.step()



