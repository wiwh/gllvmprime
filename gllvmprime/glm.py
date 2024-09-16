import torch
import torch.nn as nn
import torch.distributions as d
import numpy as np


class GLM(nn.Module):
    # this is the GLM, including the


# Distributions class should have  a .sample, .link, .inv_link
class BaseCGLM(nn.Module):
    # This is the base Conditional GLM (Conditional on the linear parameter).
    # NOTE: this is JUST the module AFTER the linear parameter is created. TODO: rename? maybe to ConditionalCGLM? (CCGLM)
    def __init__(self, dim, link=None, link_inv=None):
        super().__init__()
        self.dim = dim
        self.link = link if link is not None else self.link_default
        self.link_inv = link_inv if link_inv is not None else self.link_inv_default
        
    def forward(self, x):
        return self.mean(x)

class GaussianCGLM(BaseCGLM):
    def __init__(self, dim, link=None, link_inv=None, scale_init=1, learn_scale=True):
        super().__init__(
            dim=dim, 
            link=link, 
            link_inv=link_inv)
        
        scale = torch.ones(dim) * scale_init

        if learn_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer('scale', scale)

    def link_default(self, x):
        return x
    
    def link_inv_default(self, x):
        return x
    
    def mean(self, linpar):
        return linpar

    def sample(self, mean, *args):
        dist = d.Normal(loc=self.link_inv(mean), scale=self.scale)
        samples = dist.sample(*args)
        return samples
        
    

class BinomialCGLM(BaseCGLM):
    def __init__(self, dim, num_tries, link=None, link_inv=None):
        super().__init__(
            dim=dim, 
            link=link, 
            link_inv=link_inv)
        
        if not isinstance(num_tries, torch.Tensor):
            num_tries = torch.tensor(num_tries)
        
        if num_tries.shape[0] != self.dim:
            raise ValueError(f'num_tries must be of dimension {self.dim}, but got {num_tries.shape[0]}')

        self.register_buffer('num_tries', num_tries)
    
    def link_default(self, x):
        return torch.logit(x)
    
    def link_inv_default(self, x):
        return torch.sigmoid(x)

    def mean(self, linpar):
        probs = self.link_inv(linpar)
        
        mean = probs * self.num_tries
        return mean
    
    def sample(self, mean, *args):
        dist = d.Binomial(total_count = self.num_tries, probs=mean/self.num_tries)
        samples = dist.sample(*args)
        return samples

class PoissonCGLM(BaseCGLM):
    def __init__(self, dim, link=None, link_inv=None):
        super().__init__(
            dim=dim, 
            link=link, 
            link_inv=link_inv)
    
    def link_default(self, x):
        # Canonical log link function for Poisson GLM
        return torch.log(x)
    
    def link_inv_default(self, x):
        # Inverse link function (exponential)
        return torch.exp(x)

    def mean(self, linpar):
        # The mean is given by the inverse link function
        mean = self.link_inv(linpar)
        return mean
    
    def sample(self, mean, *args):
        # Poisson distribution with rate parameter = mean
        dist = d.Poisson(rate=mean)
        samples = dist.sample(*args)
        return samples


class CGLMEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f1 = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, y, x):
        pass


