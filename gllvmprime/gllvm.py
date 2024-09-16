import torch
import torch.nn as nn
import torch.distributions as d
from typing import List, Dict, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from glm import *

class MultiCGLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.glms = nn.ModuleList()
        self.ids = []
        self.valid = False
    
    def register_glm(self, id: np.ndarray, glm: BaseCGLM) -> None:
        # add response group to the gllvm
        assert isinstance(glm, BaseCCGLM)

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
        
        self.wz = nn.Parameter(torch.randn((self.q, self.p)))
        self.wx = nn.Parameter(torch.randn((self.k, self.p)))
        self.bias = nn.Parameter(torch.zeros(self.p)) if bias else None
        
        self.multiglm = MultiCGLM(dim=num_response)
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
        
        _, mean = self(z)
        y = self.multiglm.sample(mean)
        
        return z, y
 

 

class BasicEncoder(nn.Module):
    def __init__(self, num_latent, num_response):
        super().__init__()
        self.q = num_latent
        self.p = num_response

        self.f = nn.Sequential(
            nn.Linear(self.p, self.q),
            nn.ReLU(),
            nn.Linear(self.q, self.q),
            nn.ReLU(),
            nn.Linear(self.q, self.q),
            nn.ReLU(),
            nn.Linear(self.q, self.q)
        )
    
    def forward(self, y):
        z = self.f(y)
        return z

    
class GLLVMEncoder(nn.Module):
    def __init__(self, m):
        """m is a GLLVM instance"""
        super().__init__()
        self.m = m
        self.p = self.m.p
        self.q = self.m.q
        self.k = self.m.k

        # we concatenate, for each p, the observation, the q loadings, as well as the offset (k @ m.wx + m.bias), for total dim 1 + q + 1 = q+2

        dim_in = self.q + 2
        dim_out = self.q**2 # (this allows, for instance for the linear case, to recreate the sufficient statistics for beta)
        
        self.f1 = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )
        # now we average acrross the p dimensions and continue transforming
        self.f2 = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, self.q),
            nn.ReLU(),
            nn.Linear(self.q, self.q),
            nn.ReLU(),
            nn.Linear(self.q, self.q)
        )

    def forward(self, y, x=None):
        # y is the data, of dimension nxp each element in the 2nd dimension goes through self.f1. then we sum across p

        with torch.no_grad():
            offset = torch.zeros_like(y)
            if x is not None:
                    offset += x @ self.m.wx
            if self.m.bias is not None:
                    offset += self.m.bias
            wz = self.m.wz.data.T.unsqueeze(0).expand(y.shape[0], -1, -1)

        #print (f'wz dimension: {wz.shape}')
        #print (f'y dimension: {y.shape}')
        #print (f'offset dimension: {offset.shape}')

        input = torch.cat([y.unsqueeze(2), offset.unsqueeze(2), wz], dim = 2)
        # print(f'input dimensions: {input.shape}')
        x = self.f1(input)
        # print(f'out f1 dimensions: {x.shape}')
        
        # mean across p (dim 1)
        x = x.mean(dim=1)
        # print(f'dimension after summing: {x.shape}')

        # apply f2

        # out = self.f2(x) TODO
        
        out=x

        # print(f'dimension after f2: {out.shape}')
        
        return out

class GLLVMEncoder2(nn.Module):
    def __init__(self, m):
        """m is a GLLVM instance"""
        super().__init__()
        self.m = m
        self.p = self.m.p
        self.q = self.m.q
        self.k = self.m.k

        # we concatenate, for each p, the observation, the q loadings, as well as the offset (k @ m.wx + m.bias), for total dim 1 + q + 1 = q+2

        dim_in = self.q + 2
        dim_out = self.q**2 # (this allows, for instance for the linear case, to recreate the sufficient statistics for beta)
        
        self.f1 = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Sigmoid(),
            nn.Linear(dim_in, dim_out)
        )
        # now we average acrross the p dimensions and continue transforming
        self.f2 = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.Sigmoid(),
            nn.Linear(dim_out, self.q)
        )

    def forward(self, y, x=None):
    # y is the data, of dimension nxp each element in the 2nd dimension goes through self.f1. then we sum across p

        offset = torch.zeros_like(y)
        if x is not None:
            offset += x @ self.m.wx.detach()  # Detach only this part
        if self.m.bias is not None:
            offset += self.m.bias.detach()  # Detach only this part
        
        wz = self.m.wz.detach().T.unsqueeze(0).expand(y.shape[0], -1, -1)  # Detach only this part

        # Concatenate input
        input = torch.cat([y.unsqueeze(2), offset.unsqueeze(2), wz], dim=2)
        
        # Forward pass through f1 and f2
        x = self.f1(input)
        x = x.mean(dim=1)
        out = self.f2(x)

        return out


if __name__ == "__main__":
    #%% Example MultiCGLM

    # Instantiate the MultiCGLM (dimension 10)
    m = MultiCGLM(dim=10)
    # Instantiate glm response 1-5
    binomial = BinomialCGLM(dim=5, num_tries=[1]*5)
    # Instantiate glm responses 6-10
    poisson = PoissonCGLM(dim=3)
    gaussian = GaussianCGLM(dim=2)

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


    #%% Example GLLVM
    # --------------------
    n = 1000
    p = 10
    q = 1

    m = GLLVM(
        num_latent = q,
        num_response = p,
        num_covar = 0
    )

    gaussian = GaussianCGLM(dim=p)
    m.register_glm(np.arange(p), gaussian)
    # split = np.split(np.arange(p), 2)

    # # Instantiate glm response 
    # binomial = BinomialCGLM(dim=split[0].size, num_tries=[1]*split[0].size)
    # # Instantiate glm responses 
    # gaussian = GaussianCGLM(dim=split[1].size)


    # # Register the glms to the GLLVM
    # m.register_glm(split[0], binomial)
    # #m.register_glm(np.arange(5,8), poisson)
    # m.register_glm(split[1], gaussian)


    # %%
    # e = BasicEncoder(num_latent=q, num_response=p)
    e = GLLVMEncoder(m)
    # e = GLLVMEncoder2(m)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(e.parameters(), lr=.01)
    # optimizer = torch.optim.Adam(
    #     [param for name, param in e.named_parameters() if not name.startswith('m.')]
    # )
    # %% loop

    with torch.no_grad():
        z, y = m.sample(100)

    epochs = 1000


    for name, param in e.named_parameters():
        print(name, param)  # Inspect if gradients are non-None and updating
    
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        zhat = e(y)
        l = mse(z, zhat)
        l.backward()
        optimizer.step()
        #for name, param in e.named_parameters():
            #print(name, param.grad)  # Inspect if gradients are non-None and updating
        # print(next(e.parameters())[:10])
        # for name, param in e.named_parameters():
        #     print(name, param)  # Inspect if gradients are non-None and updating
        print(f'epoch {epoch}, loss = {l.item():.2f}')
        # plt.scatter(z.detach().cpu(), zhat.detach().cpu())

    for name, param in e.named_parameters():
        print(name, param)  # Inspect if gradients are non-None and updating

    # %%
    import matplotlib.pyplot as plt

    plt.scatter(z.detach().cpu(), zhat.detach().cpu())
    # %%
