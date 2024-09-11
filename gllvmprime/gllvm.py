import torch
import torch.nn as nn
import torch.distributions as d
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
import numpy as np



# Distributions class should have  a .sample, .link, .inv_link
class BaseGLM(nn.Module):
    def __init__(self, dim, link=None, link_inv=None):
        super().__init__()
        self.dim = dim
        self.link = link if link is not None else self.link_default
        self.link_inv = link_inv if link_inv is not None else self.link_inv_default
        
    def forward(self, x):
        return self.mean(x)

class GaussianGLM(BaseGLM):
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
        
    

class BinomialGLM(BaseGLM):
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

class PoissonGLM(BaseGLM):
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


class MultiGLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.glms = nn.ModuleList()
        self.ids = []
        self.valid = False
    
    def register_glm(self, id: np.ndarray, glm: BaseGLM) -> None:
        # add response group to the gllvm
        assert isinstance(glm, BaseGLM)

        self.glms.append(glm)
        self.ids.append(id)

    def test_valid(self):
        total_ids = np.concatenate(self.ids)
        if len(set(total_ids)) != self.dim:
            raise ValueError(f"GLMs do not cover all {self.dim} dimensions or some IDs are duplicated.")
        self.valid = True

    def forward(self, x):
        if not self.valid:
            self.test_valid()
        
        means = torch.empty_like(x)
        for id, glm in  zip(self.ids, self.glms):
            means[:,id] = glm.mean(x[:,id])
            
        return means

    def sample(self, mean=None, linpar=None, return_mean=False):
        if mean is None and linpar is None:
            raise ValueError('One of "mean" and "linpar" must be provided.')
        
        if mean is None:
            mean = self(linpar)

        sample = torch.empty_like(mean)
        
        for id, glm in zip(self.ids, self.glms):
            sample[:,id] = glm.sample(mean[:,id])
        
        return sample


class GLLVM(nn.Module):
    def __init__(self, num_latent, num_response, num_covar, bias=True):
        super().__init__()
        self.q = num_latent
        self.p = num_response
        self.k = num_covar
        
        self.wz = nn.Parameter(torch.randn((self.q, self.p))*0.1)
        self.wx = nn.Parameter(torch.randn((self.k, self.p))*0.1)
        self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None
        
        self.multiglm = MultiGLM(dim=num_response)
        self.register_glm = self.multiglm.register_glm
            
    def forward(self, z, x = None):
        """
        Compute the conditional mean of a GLLVM

        Parameters:
            - z: the latent variables, of shape (num_obs, num_latent)
            - x: covariates. If not None, must be of shape (num_obs, num_covar)
        """
        linpar = self.compute_linpar(z, x)
        mean = self.multiglm(linpar)

        return linpar, mean
    
    def compute_linpar(self, z, x):
        
        # Input checks
        if z.shape[1] != self.q:
            raise ValueError(f"Expected to receive latent variables z of shape (num_obs, {self.q}).")

        if x is None and self.k > 0:
            raise ValueError(f"Expected to receive covariates x but received None instead.")

        # Computing linpar
        # ----------------
        linpar = z @ self.wz

        if x is not None:
            linpar += x @ self.wx

        if self.bias is not None:
            linpar += self.bias
        
        return linpar
        
    
    def sample(self, num_obs=None, z=None):
        if z is None:
            z = torch.randn((num_obs, self.q), device=next(self.parameters()).device)
        
        linpar, mean = self(z)
        return self.multiglm.sample(mean)
    

#%% Example MultiGLM

# Instantiate the MultiGLM (dimension 10)
m = MultiGLM(dim=10)
# Instantiate glm response 1-5
binomial = BinomialGLM(dim=5, num_tries=[1]*5)
# Instantiate glm responses 6-10
poisson = PoissonGLM(dim=3)
gaussian = GaussianGLM(dim=2)

# Register the glms to the GLLVM
m.register_glm(np.arange(5), binomial)
m.register_glm(np.arange(5,8), poisson)
m.register_glm(np.arange(8,10), gaussian)

# example linpar
linpar = torch.rand(10,10)
# compute means
means = m(linpar)
# sample from the gllvm
m.sample(means)

# test to go to gpu
m.to("cuda")
m(linpar.to("cuda"))
m.sample(means.to("cuda"))

m.to("cpu")

#%% Example GLLVM
n = 100
p = 650000
q = 10

m = GLLVM(
    num_latent = q,
    num_response = p,
    num_covar = 0
)

split = np.split(np.arange(p), 2)

# Instantiate glm response 
binomial = BinomialGLM(dim=split[0].size, num_tries=[1]*split[0].size)
# Instantiate glm responses 
gaussian = GaussianGLM(dim=split[1].size)


# Register the glms to the GLLVM
m.register_glm(split[0], binomial)
#m.register_glm(np.arange(5,8), poisson)
m.register_glm(split[1], gaussian)

#%%
m.to("cpu")
for i in np.arange(10):
    print(i)
    m.sample(num_obs=n)

# %%
m.to("cuda")
for i in np.arange(10):
    print(i)
    m.sample(num_obs=n)

# %%
