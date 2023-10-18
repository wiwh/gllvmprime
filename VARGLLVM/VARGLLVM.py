import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import numpy as np
from VAR1 import VAR1
from typing import Type, Optional, Tuple
import matplotlib.pyplot as plt
import warnings

class VARGLLVM(nn.Module):
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
    
    def __init__(self, num_var, num_latent, num_covar, response_types, add_intercepts = True, VAR1_intercept=False, VAR1_slope=False, fixed_first_loading=False, linpred_max = 20, logvar_z1=None):
        super().__init__()
        self.response_types =  response_types
        self.response_link = {
            'bernoulli' : lambda x: torch.logit(x),
            'ordinal': lambda x: torch.logit(x),
            'poisson': lambda x: torch.log(x)
        }
        self.response_linkinv = {
            'bernoulli': lambda x: 1/(1+torch.exp(-x)),
            'ordinal': lambda x: 1/(1+torch.exp(-x)),
            'poisson': lambda x: torch.exp(x)
        }
        self.transform_functions = {
            'bernoulli' : lambda x: 2*x - 1,
            'ordinal': lambda x: 2*x - 1,
            'poisson': lambda x: torch.log(x+1)
        }

        self.num_var = num_var
        self.num_latent = num_latent
        self.num_covar = num_covar
        self.linpred_max = linpred_max

        # Define Parameters
        # -----------------
        # Parameters for the VAR
        if logvar_z1 is None:
            self.logvar_z1 = nn.Parameter(torch.zeros((num_latent,)))
        else:
            self.logvar_z1 = nn.Parameter(logvar_z1)
            self.logvar_z1.requires_grad = False
            
        self.A = nn.Parameter(torch.zeros((num_latent, num_latent)))
        if VAR1_intercept:
            self.VAR1_intercept = nn.Parameter(torch.zeros(num_latent))
        else:
            self.VAR1_intercept = nn.Parameter(torch.zeros(num_latent))
            self.VAR1_intercept.requires_grad = False
        if VAR1_slope:
            self.VAR1_slope = nn.Parameter(torch.zeros(num_latent))
        else:
            self.VAR1_slope = nn.Parameter(torch.zeros(num_latent))
            self.VAR1_slope.requires_grad = False

        # Parameters for the outcome model
        if add_intercepts:
            self.intercepts = nn.Parameter(torch.zeros((num_var,)))
        else:
            self.intercepts = None

        self.wz = nn.Parameter(torch.randn((num_latent, num_var))*0.1)
        self.wx = nn.Parameter(torch.randn((num_covar, num_var))*0.1)

        # Parameters for the random effects
        self.logvar_u = nn.Parameter(torch.zeros((num_var,)))
        
        # Define Modules
        # --------------
        self.VAR1 = VAR1(A=self.A, logvar_z1 = self.logvar_z1, beta_0 = self.VAR1_intercept, beta_1 = self.VAR1_slope)

        # Identification matters:
        # -----------------------

        self.fixed_first_loading = fixed_first_loading
        if self.fixed_first_loading:
            # set the diagonal loadings to 1
            self.loading_mask = torch.eye(self.num_latent, self.num_var).bool()
            self.wz.data = self.wz.data - (self.loading_mask * self.wz.data) + self.loading_mask
            # Register backward hook that prevents updating these
            self.wz.register_hook(self.zero_grad_hook)

    def zero_grad_hook(self, grad):
        # Set the gradient of the diagonal loadings to 1
        grad[self.loading_mask] = 0.
        return grad

    def forward(self, epsilon, u, x = None):
        """
        Compute the conditional mean of a VARGLLVM

        Parameters:
            - epsilon: shocks for the VAR
            - u: shocks for the random effects
            - x: covariates
        """
        assert epsilon.shape[2] == self.num_latent, "bad shape for epsilon"
        assert u.shape[1:] == (1, self.num_var), "bad shape for u"
        device = next(self.parameters()).device 
        # Computing linpar
        # ----------------
        linpar = torch.zeros((epsilon.shape[0], epsilon.shape[1], self.num_var)).to(device)

        # add intercepts, one per variable
        if self.intercepts is not None:
            linpar += self.intercepts[None, None, :]

        # add covariates' effects
        if x is None:
            assert self.num_covar == 0, f'VARGLLVM module expected {self.num_covar} covariates, received {0}.'
        else:
            assert(x.shape[1:] == (epsilon.shape[1], self.num_covar))
            linpar += x @ self.wx

        # add latent variables' effects
        z = self.VAR1(epsilon)

        linpar += z @ self.wz

        # finally, add random effects
        linpar += u * torch.sqrt(torch.exp(self.logvar_u[None, None, :])) # we add a time dimension: u is the same across time!
        
        linpar = linpar.clamp(-self.linpred_max, self.linpred_max)
        # compute the conditional mean
        condmean = self.linpar2condmean(linpar)

        return (linpar, condmean)
    
    def sample(self, batch_size, seq_length,  x = None, epsilon = None, u = None):
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

        if epsilon is None:
            epsilon = torch.randn((batch_size, seq_length, self.num_latent)).to(device)
        if u is None:
            u = torch.randn((batch_size, 1, self.num_var)).to(device) # one per var, but constant across time
        
        linpar, condmean = self(epsilon, u, x)
        y = self.sample_response(condmean)

        return {"epsilon": epsilon, "u": u, "linpar": linpar, "condmean":condmean, "y":y, "x":x}

    def sample_response(self, mean):
        device = next(self.parameters()).device
        y = torch.zeros_like(mean).to(device)
        for response_type, response_id in self.response_types.items():
            if response_type == "bernoulli":
                y[:,:,response_id] = torch.bernoulli(mean[:,:,response_id]).to(device)
            elif response_type == "ordinal":
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
        for response_type, response_id in self.response_types.items():
            mean[:,:,response_id] = self.response_linkinv[response_type](linpar[:,:,response_id])
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

class MELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, linpar, ys, linpars, mask=None):
        """Computes the loss. hat is recontructed y, ys is simulated"""
        if mask is not None:
            loss = -torch.sum((y * linpar - ys * linpars) * ~mask )/ y.shape[0]
        else: 
            loss = -torch.sum(y * linpar - ys * linpars) / y.shape[0] 
        return loss


class Encoder(nn.Module):
    """
    Learns the shocks epsilon and u of a VARGLLM, such that when passed through VARGLLM we get the reconstruction.
    
    Parameters:
        - num_var: number of responses
        - num_covar: number of covariates (columns of x)
        - num_latent: dimension of latent variables (for a single period)
        - num_hidden: number of hidden units. I would advise fully connected except for large number of covar
    """
    def __init__(self, num_var: int, num_covar: int, num_latent: int, num_hidden: int, transform=True):
        super().__init__()
        
        self.num_var = num_var
        self.num_covar = num_covar
        self.num_latent = num_latent
        self.transform = transform

        self.fc = nn.Sequential(
            nn.Linear(num_var + num_covar, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            # nn.Linear(num_hidden, num_hidden),
            # nn.Tanh(),
            nn.Linear(num_hidden, num_latent + num_var)
        )

    def forward(self, y: torch.Tensor, VARGLLVM_model: Type[nn.Module], x: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict z, u given y,x, and a VARGLLVM_model

        Parameters:
            y: a (batch_size, seq_length, num_var) Tensor of responses
            x: a (batch_size, seq_length, num_covar) Tensor of covariates
            VARGLLVM_model: a VARGLLVM model in eval mode.
        """
        
        # Flag VARGLLVM_model
        # -------------------
        
        # First we set the model in eval mode
        VARGLLVM_model.eval()
        # We need to turn off the requires_grad of all VARGLLVM_model parameters, and then restore them to their original state
        # Store the original requires_grad flags
        # TODO: do we reall need to do this here? Do it outside imho.
        original_flags = [p.requires_grad for p in VARGLLVM_model.parameters()]

        # Set requires_grad to False for all parameters of VARGLLVM_model
        for param in VARGLLVM_model.parameters():
            param.requires_grad = False
        


        # Forward pass
        # ------------
        if self.transform:
            with torch.no_grad():
                y = VARGLLVM_model.transform_responses(y)
        if x is None:
            xy = y
        else:
            xy = torch.cat((y, x), dim=2)
        
        seq_length = y.shape[1]
        out = self.fc(xy.view(-1, self.num_var + self.num_covar)).view(-1, seq_length, self.num_latent + self.num_var)

        out_z = out[:, :, :self.num_latent]
        out_epsilon = VARGLLVM_model.VAR1.backward(out_z)
        
        out_u_scaled = out[:, :, self.num_latent:]
        out_u = out_u_scaled / torch.sqrt(torch.exp(VARGLLVM_model.logvar_u[None, None, :]))
        out_u = torch.mean(out_u, dim=1).unsqueeze(1)
        
        # Set Flags of VARGLLVM_model back
        # ---------------------------------
        # Restore the original requires_grad flags
        for param, flag in zip(VARGLLVM_model.parameters(), original_flags):
            param.requires_grad = flag


        return (out_epsilon, out_u)
    

def impute_values(model, encoder, y, mask, x=None, impute_with=None, keep_within_range=True):
    """
    Imput the missing values: returns y with all missing values marked as True in the mask is imputed.

    Parameters:
        - model: the VARGLLVM model
        - mask: a boolean tensor like y with True if missing and False otherwise
        - impute_with: if epsilon or u are missing, impute the missing value with the value in impute_with, otherwise encode/decode
        - keep_within_range: for all variables, keeps the imputed values within the range of the observed values.
    """

    if impute_with is not None:
        y[mask] = impute_with
    else:
        epsilon, u = encoder(y, model, x=x)
        _, condmean = model(epsilon, u, x)
        y[mask] = condmean[mask]

    if keep_within_range:

        y_max_observed = y[~mask].max(dim=0)[0].max(dim=0)[0]
        y_min_observed = y[~mask].min(dim=0)[0].min(dim=0)[0]
        y = torch.clamp(y, y_min_observed, y_max_observed)

    return y
    
    
def train_encoder(encoder, VARGLLVM_model, criterion, optimizer, num_epochs=100, sample=True, batch_size=None, seq_length =None, x=None, data=None, verbose=False, mask=None, impute_with=None):
    """
    Trains the given encoder model using the provided data and parameters.

    Parameters:
    - encoder: The encoder model to be trained.
    - model: VARGLLVM model for response transformation.
    - data_true: Ground truth data dictionary with 'y' and 'epsilon' keys.
    - x: Input data.
    - num_epochs (optional): Number of epochs for training.
    - lr (optional): Learning rate for the Adam optimizer.
    - num_hidden (optional): Number of hidden units.
    - num_covar (optional): Number of covariates.
    - num_latent (optional): Number of latent variables.
    - data: if sample is False, takes data from data
    
    Returns:
    - A trained encoder.
    """
    encoder.train()
    VARGLLVM_model.eval()


    if data is not None and sample:
        print("Sample is True: supplied 'data' is ignored.")

    num_latent = VARGLLVM_model.wz.shape[0]
    num_var  = VARGLLVM_model.wz.shape[1]
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        if sample:
            with torch.no_grad():
                data = VARGLLVM_model.sample(batch_size, seq_length, x=x)
                if mask is not None:
                    data['y'] = impute_values(VARGLLVM_model, encoder, data['y'], mask, data['x'], impute_with=impute_with)  

        epsilon, u = encoder(data['y'], VARGLLVM_model=VARGLLVM_model, x=data['x'])
        
        tot_weight = num_var + num_latent
        loss = criterion(epsilon, data['epsilon']) * num_latent/tot_weight + criterion(u, data['u']) * tot_weight/tot_weight
        
        loss.backward()
        optimizer.step()
        if verbose:
            print(f'Epoch {epoch}, loss {loss.item()}.')

    return loss.item()

def train_decoder(model, encoder, criterion, optimizer, data, num_epochs=100, transform=False, verbose=True, clip_value=0.05, mask=None, impute_with=None):
        batch_size = data['y'].shape[0]
        seq_length = data['y'].shape[1]

        # Training loop
        model.train()
        encoder.eval()
        # We start by training the encoder

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            with torch.no_grad():
                # # simulate
                data_sim = model.sample(batch_size, seq_length, data['x'])

                # impute if necessary
                if mask is not None:
                    data['y'] = impute_values(model, encoder, data['y'], mask, data['x'], impute_with=impute_with)
                    data_sim['y'] = impute_values(model, encoder, data_sim['y'], mask, data['x'], impute_with=impute_with)
                    # TODO (remove) print(f"NA in data: {torch.isnan(data['y']).sum()}, {torch.isnan(data_sim['y']).sum()}")

                # encode
                data_epsilon, data_u = encoder(data['y'], model, data['x'])
                data_sim_epsilon, data_sim_u = encoder(data_sim['y'], model, data_sim['x'])

            # TODO (remove) print(f"has na? epsilon: {torch.sum(torch.isnan(data_epsilon))}, u: {torch.sum(torch.isnan(data_u))}")
            
            linpar, _ = model(data_epsilon, data_u, data['x'])
            linpar_sim, _ = model(data_sim_epsilon, data_sim_u, data['x'])

            # TODO (remove) print(f"max_linpar: {torch.max(torch.abs(linpar))}")
            
            if transform:
                with torch.no_grad():
                    y = model.transform_responses(data['y']) 
                    y_sim = model.transform_responses(data_sim['y'])
            else :
                y = data['y']
                y_sim = data_sim['y']
            loss = criterion(y, linpar, y_sim, linpar_sim, mask)  # TODO add mask
            
            loss.backward()
            clip_grad_value_(model.parameters(), clip_value=clip_value)
            optimizer.step()
            
            
            # Check if it is stationary
            A = model.A
            if torch.linalg.matrix_norm(A, ord=2) >= 0.95:
                with torch.no_grad():
                    warnings.warn("||A||_2 >=0.95: VAR is potentially nonstationary, norm set to 0.95 instead.")
                    
                    # Compute the SVD
                    U, S, V = torch.linalg.svd(A)
                    
                    # Adjust the largest singular value
                    S[0] = 0.95
                    
                    # Reconstruct the matrix A
                    A_adjusted = U @ torch.diag(S) @ V.T

                    # Assign back to A or use A_adjusted as needed
                    A[:] = A_adjusted


            if verbose: 
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
            
        return loss.item()