# Module for Multiple Generalized Linear Model. Generate multiple conditional means, and also has a import torch
import torch.nn as nn
import torch.distributions as d
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class GLMFamilies:
    p: int  # num_var
    q: int  # num_latent
    k: int  # num_covar
    family: Dict[str, List[int]]
    args: Dict[str, List] = field(default_factory=lambda: {
        'binomial': {'size':[1]}
    })
    link: Dict[str, Callable] = field(default_factory=lambda: {
        'gaussian': lambda x: x,
        'binomial': torch.logit,
        'ordinal': torch.logit,
        'poisson': torch.log
    })
    linkinv: Dict[str, Callable] = field(default_factory=lambda: {
        'gaussian': lambda x: x,
        'binomial': lambda x: 1 / (1 + torch.exp(-x)),
        'ordinal': lambda x: 1 / (1 + torch.exp(-x)),
        'poisson': torch.exp
    })
    transform: Dict[str, Callable] = field(default_factory=lambda: {
        'gaussian': lambda x: x,
        'binomial': lambda x: 2 * x - 1,
        'ordinal': lambda x: 2 * x - 1,
        'poisson': torch.log1p
    })
    
class GLLVM(nn.Module):
    """
    Vector Autoregressive Generalized Linear Latent Variable Model (VAR GLLVM) for multivariate longitudinal data.
    
    This class provides a multivariate extension of GLLVM, modeling multivariate responses over time.
    The variability in the data is captured through three sources: cross-sectional associations,
    cross-lagged associations, and associations between repeated measures of the same response over time.
    The model specification integrates a time-dependent latent variable and an item-specific random effect.
    
    Parameters:
    - num_var: int, Number of response variables.
    - num_latent: int, Number of latent variables.
    - num_covar: int, Number of observed covariates.
    - response_types: dict, Mapping of response type to its indices.
    - add_intercepts: bool, Whether to include intercepts in the model.
    - fixed_first_loading: bool, Whether to fix to 1. the first loading.
    
    Key Methods:
    - forward: Computes the conditional mean of the model.
    - sample: Draws samples from the VARGLLVM model.
    - sample_response: Samples from the response distribution based on conditional mean.
    - linpar2condmean: Converts linear predictors to conditional means.

    Model Specification:
    y_{ijt}|μ_{ijt} ~ F_j(y_{ijt}|μ_{ijt}, τ_j)
    μ_{ijt} = g_j(β_{0jt} + x_{it}^T*β_{jt} + z_{it}^T*λ_{jt}+u_{ij}*σ_{u_j})
    
    Where:
    - g_j: link function
    - μ_{ijt}: mean of the distribution
    - F_j: a distribution from the exponential family
    - η_{ijt}: linear predictor

    Temporal evolution of the latent variable:
    z_{it} = A*z_{i,t-1} + ε_{it}
    with ε_{it} ~ N(0, I) and initialization z_{i, 1} ~ N(0, Σ_{z1}).
    
    The random effects are assumed independent of the latent variable and distributed as:
    u_{i} ~ N_p(0, I).
    """
    
    def __init__(self, input_size, output_size, families=None, response_args=None, response_link=None, response_linkinv=None, bias=True):
        super().__init__()
        self.settings = GLLVMSettings(
            p=num_var,
            q=num_latent,
            k=num_covar,
            family=response_family,
            args=response_args,
            link=response_link,
            linkinv=response_linkinv
        )

        # Define Parameters
        # -----------------
        self.wz = nn.Parameter(torch.randn((num_latent, num_var))*0.1)
        self.wx = nn.Parameter(torch.randn((num_covar, num_var))*0.1)
        self.log_scale = nn.Parameter(torch.zeros((num_covar)))

        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_var)))
        
    def forward(self, z, x = None):
        """
        Compute the conditional mean of a GLLVM

        Parameters:
            - z: the latent variables, of shape (num_obs, num_latent)
            - x: covariates. If not None, must be of shape (num_obs, num_covar)
        """
        
        # Input checks
        if z.shape[1] != self.settings.q:
            raise ValueError(f"Expected to receive latent variables z of shape (num_obs, {self.settings.q}).")

        if x is None and self.settings.k > 0:
            raise ValueError(f"Expected to receive covariates x but received None instead.")

        if x is not None and ((x.shape[0] != z.shape[0]) or (x.shape[1] != self.settings.k)):
            raise ValueError(f"Expected to receive covariates x of shape ({z.shape[0], self.settings.k}) but is of shape {x.shape} instead.")

        device = next(self.parameters()).device 

        # Computing linpar
        # ----------------
        linpar = torch.zeros((z.shape[0]), device=device)

        if self.has_bias:
            linpar += self.bias
        if x is not None:
            linpar += x @ self.wx

        linpar += z @ self.wz

        # linpar = linpar.clamp(-self.linpred_max, self.linpred_max) # TODO:clamp if necessary

        # compute the conditional mean
        condmean = self.linpar2condmean(linpar)

        return (linpar, condmean)
    
    def sample(self, z, x = None):
        """
        Sample from the VARGLLVM. 

        Parameters:
            - batch_size: number of observational units
            - seq_length: length of the sequence. if any of x, epsilon, or u are provided, their seq_length must coincide
            - x: tensor of shape (batch_size, seq_length, num_covar). If the VARGLLVM model was initialized with num_covar >= 1, cannot be None.
            - epsilon: the shocks for the latent variables of shape (batch_size, seq_length, num_latent). If None, those are drawn iid from N(0, 1).
            - u: the shocks for the random effects of shape (batch_size, 1, num_var). If None, those are drawn iid from N(0, 1).

        """
        device = next(self.parameters()).device

        _, mean = self(z, x)

        y = torch.zeros_like(mean).to(device)
        for response_type, response_id in self.response_types.items():
            if response_type == "gaussian":
                y[:,:,response_id] = torch.randn((z.shape[0], len(response_id)) * torch.exp(self.log_scale[response_id]) + mean[:,response_id], device=device)
            elif response_type == "binomial":
                binomial = d.binomial.Binomial(total_count = self.settings['args']['binomial']['size'])
                y[:,:,response_id] = torch.binomial(mean[:,:,response_id], device=device)
            elif response_type == "ordinal":
                # TODO: check the logic of ordinal
                cum_probs = mean[:,:,response_id]
                # draw one uniform for the whole vector
                random = torch.rand((*cum_probs.shape[0:2], 1)).to(device)
                # compare with the cumulative probabilities
                ordinal = torch.sum(random > cum_probs, dim=2)
                ordinal = torch.nn.functional.one_hot(ordinal).squeeze().float()
                ordinal = ordinal[:,:,1:] # discard the first column of the one_hot encoding, as it is superfluous (as a 0)
                y[:,:,response_id] = ordinal
            elif response_type == "poisson":
                y[:,:,response_id] = torch.poisson(mean[:,:,response_id])

        return y


    def linpar2condmean(self, linpar):
        mean  = torch.zeros_like(linpar)
        for family, response_id in self.family.items():
            if family == "binomial":
                mean[:,:,response_id] = self.response_linkinv
            else:
                mean[:,:,response_id] = self.response_linkinv[family](linpar[:,:,response_id])
            
        return mean

    
    def transform_responses(self, y, transform_functions=None):
        """
        Transform the responses according to their types.

        Parameters:
            - y: a (batch_size, seq_length, num_var) tensor
            - transform_functions: a dictionary with keys:values corresponding to response_type: lambda x: transform(x). Defaults to the default transforms of VARGLLVM
        """

        if transform_functions is None:
            transform_functions = self.transform_functions

        y_transformed = torch.zeros_like(y)
        for response_type, response_id in self.response_types.items():
            y_transformed[:,:,response_id] = self.transform_functions[response_type](y[:,:,response_id])

        return y_transformed

    
    def _get_device(self):
        return next(self.parameters()).device
