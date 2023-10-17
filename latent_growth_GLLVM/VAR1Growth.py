import torch
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import warnings

class VAR1(nn.Module):
    def __init__(self, A, logvar_z1, beta_0=None, beta_1 = None):
        """
        Initialize the VAR1 model.

        Parameters:
        - A (torch.Tensor): Matrix of auto-correlations. Should be square.
        - logvar_z1 (torch.Tensor): Element-wise log of the diagonal elements 
          of Sigma_z1 for initialization. We learn the log to ensure 
          the values, when exponentiated, are always positive.

        Note: The dimension of A and logvar_z1 determines the dimension of the autoregression.
        """
        # Ensure A is square and its size matches logvar_z1's size.
        assert(A.shape[0] == logvar_z1.shape[0])
        assert(A.shape[0] == A.shape[1])
        
        super().__init__()

        if torch.linalg.matrix_norm(A, ord=2) >= 1:
          warnings.warn("||A||_2 >= 1: VAR is potentially nonstationary.")
        
        # Intercept
        if beta_0 is None:
            self.beta_0 = torch.zeros(A.shape[0])  # no grad required
        else:
            self.beta_0 = beta_0
        # Slope
        if beta_1 is None:
            self.beta_1 = torch.zeros(A.shape[0])  # no grad required
        else:
            self.beta_1 = beta_1
        self.dim = A.shape[0]
        self.A = A
        self.logvar_z1 = logvar_z1

    def forward(self, epsilon):
        """
        Forward pass for the VAR1 model.

        Parameters:
        - epsilon (torch.Tensor): Matrix of shocks. Assumes all shocks are from 
          N(0, I). The tensor shape should be (num_batch, num_seq, num_features).

        Returns:
        - z (torch.Tensor): Generated sequence following the VAR1 process. 
          Shape: (num_batch, num_seq, num_features).
        """
        # Ensure the last dimension of epsilon matches the size of A.
        assert(epsilon.shape[2] == self.A.shape[0])
        
        # Initialize a tensor to hold the generated sequence.
        z = torch.zeros_like(epsilon)
        
        # Initialize the first period. Note: unsqueeze(0) allows broadcasting across batches.
        z[:, 0, :] = epsilon[:, 0, :] * torch.sqrt(torch.exp(self.logvar_z1)).unsqueeze(0) + self.beta_0
        
        # Loop over time to apply the VAR1 process.
        for i in range(1, epsilon.shape[1]):
            z[:, i] = self.beta_0 + self.beta_1*i + (self.A @ z[:, i-1].clone().unsqueeze(-1)).squeeze(-1) + epsilon[:, i]
        
        return z
    
    def backward(self, z):
        """
        Goes backward: from z, find epsilon

        Parameters:
        - z (torch.Tensor): Generated sequence following the VAR1 process. 
        Shape: (num_batch, num_seq, num_features).

        Returns:
        - epsilon (torch.Tensor): Matrix of shocks that generated the sequence z.
        Shape: (num_batch, num_seq, num_features).
        """
        # Ensure the last dimension of z matches the size of A.
        assert(z.shape[2] == self.A.shape[0])
        
        # Initialize a tensor to hold the recovered shocks.
        epsilon = torch.zeros_like(z)
        
        # For the first period, z_0 = epsilon_0 * sqrt(exp(logvar_z1))
        epsilon[:, 0, :] = (z[:, 0, :]) / torch.sqrt(torch.exp(self.logvar_z1)).unsqueeze(0)
        
        # Loop over time to compute the shocks epsilon.
        for i in range(1, z.shape[1]):
            epsilon[:, i] = z[:, i] - self.beta_0 - self.beta_1 * i - (self.A @ z[:, i-1].unsqueeze(-1)).squeeze(-1)
            
        return epsilon


    def sample(self, num_batch, seq_length):
        """
        Sample a random AR1.

        Parameters:
        - n (int): num_batch
        - T (int): sequence length
        """
        epsilon = torch.randn((num_batch, seq_length, self.dim))
        z = self(epsilon)
        return (z, epsilon)