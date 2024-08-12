import torch
import torch.nn as nn


class VAR1(nn.Module):
  def __init__(self, dim):
      """
      Initialize the VAR1 model.

      Parameters:
      - dim: the dimension of the var1.

      - beta (torch.Tensor): the vector of intercepts (optional).

      Note: The dimension of A and logvar_z1 determines the dimension of the autoregression.
      """

      super().__init__()
      self.dim = dim
      self.L = nn.Parameter(torch.diag(torch.ones(dim)))
      self.Sigma_noise = torch.diag(torch.ones(dim)) # the noise of the 
  
  def project_spectral_norm(self, eps=1e-4):
    A = self.L.T @ self.L
    norm = torch.linalg.norm(A, ord=2)
    if norm/(1+eps) >= 1.:
      L = L / torch.sqrt(norm/(1+eps))
    return L

  def forward(self, noise):
      """
      Forward pass for the VAR1 model.

      Parameters:
      - noise (torch.Tensor): Matrix of shocks. Assumes all shocks are from 
        N(0, 1). The tensor shape should be (num_batch, num_seq, num_features).

      Returns:
      - z (torch.Tensor): Generated sequence following the VAR1 process. 
        Shape: (num_batch, num_seq, num_features).
      """
        
      # Initialize a tensor to hold the generated sequence.
      z = torch.zeros_like(noise)
      A = self.L.T @ self.L
      
      # Loop over time to apply the VAR1 process.
      z[:,0] = noise[:,0]
      for i in range(1, noise.shape[1]):
          z[:, i] = (A @ z[:, i-1].clone().unsqueeze(-1)).squeeze(-1) + noise[:, i]
      
      return z
  
  # def backward(self, z):
  #     """
  #     Goes backward: from z, find noise

  #     Parameters:
  #     - z (torch.Tensor): Generated sequence following the VAR1 process. 
  #     Shape: (num_batch, num_seq, num_features).

  #     Returns:
  #     - noise (torch.Tensor): Matrix of shocks that generated the sequence z.
  #     Shape: (num_batch, num_seq, num_features).
  #     """
  #     # Ensure the last dimension of z matches the size of A.
  #     assert(z.shape[2] == self.A.shape[0])
      
  #     # Initialize a tensor to hold the recovered shocks.
  #     noise = torch.zeros_like(z)
      
  #     # Loop over time to compute the shocks noise.
  #     for i in range(1, z.shape[1]):
  #         noise[:, i] = z[:, i] - self.beta_0 - self.beta_1 * i - (self.A @ z[:, i-1].unsqueeze(-1)).squeeze(-1)
          
  #     return noise


  def sample(self, num_batch, seq_length, init=None):
      """
      Sample a random AR1.

      Parameters:
      - n (int): num_batch
      - T (int): sequence length
      """
      noise = torch.randn((num_batch, seq_length, self.dim))
      if init is not None:
        noise[:,0] = init
      z = self(noise)
      return (z, noise)