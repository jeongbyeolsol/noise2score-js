import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nonlinear_func, expand_tensor, sample_laplace_noise, sample_unit_laplace_noise
#from models.layers import MLP, WNMLP, Identity


def add_gaussian_noise(input, std):
    eps = torch.randn_like(input)
    return input + std*eps, eps

def add_poisson_noise(input, peak=30.0, clamp=True):
    """
    Poisson noise.

    input: [0, 1] 범위의 이미지 텐서라고 가정
    peak: photon count scale. 작을수록 노이즈가 강함.
          예: 10 강함, 30 보통, 100 약함
    clamp: 결과를 [0, 1]로 자를지 여부

    return:
        noisy: 포아송 노이즈가 들어간 이미지
        eps: noisy - input, 실제로 더해진 residual noise
    """

    # Poisson rate는 음수일 수 없으므로 최소 0으로 제한
    rate = input.clamp_min(0) * peak

    # photon count 샘플링
    noisy_counts = torch.poisson(rate)

    # 다시 [0, 1] scale로 복원
    noisy = noisy_counts / peak

    if clamp:
        noisy = noisy.clamp(0, 1)

    eps = noisy - input
    return noisy, eps


def add_gamma_noise(input, concentration=2.0, clamp=True):
    """
    Gamma multiplicative noise.

    input: [0, 1] 범위의 이미지 텐서라고 가정
    concentration: 감마분포 shape parameter.
                   작을수록 노이즈가 강함.
                   예: 0.5 매우 강함, 2 보통, 10 약함

    감마 노이즈는 평균이 1이 되도록 rate=concentration으로 둠.
    즉 gamma_noise ~ Gamma(alpha, alpha)
    평균 alpha / alpha = 1

    return:
        noisy: 감마 곱셈 노이즈가 들어간 이미지
        eps: noisy - input, 실제 residual noise
    """

    alpha = torch.full_like(input, concentration)
    rate = torch.full_like(input, concentration)

    gamma_noise = torch.distributions.Gamma(alpha, rate).sample()

    noisy = input * gamma_noise

    if clamp:
        noisy = noisy.clamp(0, 1)

    eps = noisy - input
    return noisy, eps

"""
def add_uniform_noise(input, val):
    #raise NotImplementedError
    #eps = 2.*val*torch.rand_like(input) - val
    eps = torch.rand_like(input)
    return input + 2.*val*eps-val, eps
"""
"""
def add_laplace_noise(input, scale):
    eps = sample_unit_laplace_noise(shape=input.size(), dtype=input.dtype, device=input.device)
    return input + scale*eps, eps
"""

class MLP(nn.Module):
  def __init__(self, *args, **kwargs):
      super().__init__()
    
  def forward():
    pass

class ARDAE(nn.Module):
    def __init__(self,
                 input_dim=2,
                 h_dim=1000,
                 std=0.1,
                 num_hidden_layers=1,
                 nonlinearity='tanh',
                 noise_type='gaussian',
                 #init=True,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.std = std
        self.num_hidden_layers = num_hidden_layers
        self.nonlinearity = nonlinearity
        self.noise_type = noise_type
        #self.init = init

        self.main = MLP(input_dim+1, h_dim, input_dim, use_nonlinearity_output=False, num_hidden_layers=num_hidden_layers, nonlinearity=nonlinearity)

    def add_noise(self, input, std=None):
        std = self.std if std is None else std
        if self.noise_type == 'gaussian':
            return add_gaussian_noise(input, std)
        elif self.noise_type == 'poisson':
            return add_poisson_noise(input, std)
        elif self.noise_type == 'gamma':
            return add_gamma_noise(input, std)
        else:
            raise NotImplementedError

    def loss(self, input, target):
        # recon loss (likelihood)
        recon_loss = F.mse_loss(input, target)#, reduction='sum')
        return recon_loss

    def forward(self, input, std=None):
        # init
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        if std is None:
            std = input.new_zeros(batch_size, 1)
        else:
            assert torch.is_tensor(std)

        # add noise
        x_bar, eps = self.add_noise(input, std)

        # concat
        h = torch.cat([x_bar, std], dim=1)

        # predict
        glogprob = self.main(h)

        ''' get loss '''
        loss = self.loss(std*glogprob, -eps)

        # return
        return None, loss

    def glogprob(self, input, std=None):
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        if std is None:
            std = input.new_zeros(batch_size, 1)
        else:
            assert torch.is_tensor(std)

        # concat
        h = torch.cat([input, std], dim=1)

        # predict
        glogprob = self.main(h)

        return glogprob
      
      
class MLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=8,
                 output_dim=2,
                 nonlinearity='relu',
                 num_hidden_layers=1,
                 use_nonlinearity_output=False,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        layers = []
        if num_hidden_layers >= 1:
            for i in range(num_hidden_layers):
                layers += [nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(input_dim if num_hidden_layers==0 else hidden_dim, output_dim)

    def forward(self, input):
        # init
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        # forward
        hidden = x
        if self.num_hidden_layers >= 1:
            for i in range(self.num_hidden_layers):
                hidden = nn.SELU(self.nonlinearity)(self.layers[i](hidden))
        output = self.fc(hidden)
        if self.use_nonlinearity_output:
            output = nn.SELU(self.nonlinearity)(output)