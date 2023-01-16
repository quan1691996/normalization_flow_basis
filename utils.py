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

class ShapesDataset(data.Dataset):
    def __init__(self, array):
        self.array = array.astype(np.float32) / 2.0
        self.array = np.transpose(self.array, (0,3,1,2))

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

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

class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_base_point, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(include_base_point)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def create_mask(self, include_base_point):
        h_by_2 = self.kernel_size[0] // 2
        w_by_2 = self.kernel_size[1] // 2
        self.mask[:, :, :h_by_2] = 1
        self.mask[:, :, h_by_2, :w_by_2] = 1
        if include_base_point:
            self.mask[:, :, h_by_2, w_by_2] = 1

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        x = super().forward(x)
        x = x.permute(0,3,1,2).contiguous()
        return x


class AutoRegressiveFlow(nn.Module):
    def __init__(self, num_channels_input, num_layers=5, num_channels_intermediate=64, kernel_size=7, n_components=2, **kwargs):
        super(AutoRegressiveFlow, self).__init__()
        first_layer = MaskedConv2d(False, num_channels_input, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        model = [first_layer]
        block = lambda: MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)

        for _ in range(num_layers):
            model.append(LayerNorm(num_channels_intermediate))
            model.append(nn.ReLU())
            model.append(block())

        second_last_layer = MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, 1, **kwargs)
        last_layer = MaskedConv2d(True, num_channels_intermediate, n_components * 3 * num_channels_input, 1, **kwargs)
        model.append(second_last_layer)
        model.append(last_layer)

        self.model = nn.Sequential(*model)
        self.n_components = n_components

    def forward(self, x):
        batch_size, c_in = x.size(0), x.size(1) # x.size() is (B, c_in, h, w)
        h_and_w = x.size()[2:]
        out = self.model(x) # out.size() is (B, c_in * 3 * n_components, h, w)
        out = out.view(batch_size, 3 * self.n_components, c_in, *h_and_w) # out.size() is (B, 3*n_components, c_in, h, w)
        mus, log_sigmas, weight_logits = torch.chunk(out, 3, dim=1) # (B, n_components, c_in, h, w)
        weights = torch.nn.functional.softmax(weight_logits, dim=1)

        distribution = Normal(mus, log_sigmas.exp())

        x = x.unsqueeze(1) # x.size() is (B, 1, c_in, h, w)
        z = distribution.cdf(x) # z.size() is (B, n_components, c_in, h, w)
        z = (z * weights).sum(1) # z.size() is (B, c_in, h, w)

        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(1).log()

        return z, log_dz_by_dx

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AffineTransform2D(nn.Module):
    def __init__(self, left, hidden_size=64, num_hidden_layers=2):
        super(AffineTransform2D, self).__init__()
        self.mlp = MLP(2, hidden_size, num_hidden_layers, 2)
        self.mask = torch.FloatTensor([1,0]) if left else torch.FloatTensor([0,1])
        self.mask = self.mask.view(1,-1)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, reverse=False):
        # x.size() is (B,2)
        x_masked = x * self.mask
        # log_scale and shift have size (B,1)
        log_scale, shift = self.mlp(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale
        # log_scale and shift have size (B,2)
        shift = shift  * (1-self.mask)
        log_scale = log_scale * (1-self.mask)
        if reverse:
            x = (x - shift) * torch.exp(-log_scale)
        else:
            x = x * torch.exp(log_scale) + shift
        return x, log_scale


class RealNVP(nn.Module):
    def __init__(self, affine_transforms):
        super(RealNVP, self).__init__()
        self.transforms = nn.ModuleList(affine_transforms)

    def forward(self, x):
        z, log_det_jacobian = x, torch.zeros_like(x)
        for transform in self.transforms:
            z, log_scale = transform(z)
            log_det_jacobian += log_scale
        return z, log_det_jacobian

    def invert(self, z):
        for transform in self.transforms[::-1]:
            z, _ = self.transform(z)
        return z

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
    with torch.no_grad():
        for x in data_loader:
            x = x.float()
            z, log_dz_by_dx = model(x)
            loss = loss_function(target_distribution, z, log_dz_by_dx)
            total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

# This function is applied to the models immediately after initialization.
def weights_init(m):
    if isinstance(m, MaskedConv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
