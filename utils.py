import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions.normal import Normal 
import numpy as np

def generate_mixture_of_2_gaussians(num_of_points):
    n = num_of_points//2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=0.25, scale=0.5, size=(num_of_points-n,))
    return np.concatenate([gaussian1, gaussian2])

def generate_mixture_of_3_gaussians(num_of_points):
    n = num_of_points//3
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=1.5, scale=0.35, size=(n,))
    gaussian3 = np.random.normal(loc=0.0, scale=0.2, size=(num_of_points-2*n,))
    return np.concatenate([gaussian1, gaussian2, gaussian3])

class NumpuDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_by_dx

class LogitTransform(nn.Module):
    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha 

    def forward(self, x):
        x_new = self.alpha/2 + (1-self.alpha)*x 
        z = torch.log(x_new) - torch.log(1-x_new)
        log_dz_by_dx = torch.log(torch.FloatTensor([1-self.alpha])) - torch.log(x_new) - torch.log(1-x_new)
        return z, log_dz_by_dx
        

class FlowComposable1d(nn.Module):
    def __init__(self, flow_models_list):
        super(FlowComposable1d, self).__init__()
        self.flow_models_list = nn.ModuleList(flow_models_list)

    def forward(self, x):
        z, sum_log_dz_by_dx = x, 0
        for flow in self.flow_models_list:
            z, log_dz_by_dx = flow(z)
            sum_log_dz_by_dx += log_dz_by_dx
        return z, sum_log_dz_by_dx

class CDFParams(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_hidden_layers=3, output_size=None):
        super(CDFParams, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConditionalFlow1D(nn.Module):
    def __init__(self, n_components):
        super(ConditionalFlow1D, self).__init__()
        self.cdf = CDFParams(output_size=n_components*3)

    def forward(self, x, condition):
        x = x.view(-1,1)
        mus, log_sigmas, weight_logits = torch.chunk(self.cdf(condition), 3, dim=1)
        weights = weight_logits.softmax(dim=1)
        distribution = Normal(mus, log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_by_dx

class Flow2d(nn.Module):
    def __init__(self, n_components):
        super(Flow2d, self).__init__()
        self.flow_dim1 = Flow1d(n_components)
        self.flow_dim2 = ConditionalFlow1D(n_components)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, log_dz1_by_dx1 = self.flow_dim1(x1)
        z2, log_dz2_by_dx2 = self.flow_dim2(x2, condition=x1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        log_dz_by_dx = torch.cat([log_dz1_by_dx1.unsqueeze(1), log_dz2_by_dx2.unsqueeze(1)], dim=1)
        return z, log_dz_by_dx

def loss_function(target_distribution, z, log_dz_by_dx):
    log_likelihood = target_distribution.log_prob(z) + log_dz_by_dx
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        x = x.float()
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        x = x.float()
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()