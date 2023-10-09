import numpy as np
import torch
from torch.nn.utils import clip_grad_value_
from torch import nn
from sklearn.linear_model import PoissonRegressor
from pathlib import Path

class MPLN(nn.Module):
    def __init__(self, num_latents, num_features, num_covariates, V, A = None):
        super().__init__()
        self.num_latents = num_latents
        self.num_features = num_features
        self.num_covariates = num_covariates
        self.V = torch.tensor(V).float()
        self.encoder = Encoder(num_features, num_covariates, num_latents, num_hidden=100)

        # Initialize parameters
        self.B = nn.Parameter(torch.zeros((num_features, num_covariates)))
        self.a = nn.Parameter(torch.Tensor(A)) if A is not None else nn.Parameter(torch.ones(num_latents))

        # Initialize solver for W
        # Create Poisson regression model with L2 regularization
        self.poissonSolver =  PoissonRegressor(alpha=1/(num_features), fit_intercept=True) # TODO: check this intercept
    
    def to(self, *args, **kwargs):
        self.V = self.V.to(*args, **kwargs)
        super().to(*args, **kwargs)

        return self

    def forward(self, W, X=None, invlink=torch.exp):
        if X is None:
            X = torch.zeros((W.shape[0], 0)).to(self.V.device)
        # Get the linpar (Z above) from the eigendecomposition
        VA = self.V * torch.sqrt(torch.exp(self.a.unsqueeze(0)))
        linpar = X @ self.B.T + W @ VA.T
        condmean = linpar if invlink is None else invlink(linpar)
        return linpar, condmean

    def sample(self, n, X=None, poisson=True):
        with torch.no_grad():
            W = torch.randn((n, self.num_latents)).to(self.V.device)

            if poisson is True:
                _, condmean = self(W, X, invlink=torch.exp)
                Y = torch.poisson(condmean).float()
            else:
                _, Y = self(W, X)

        return {'W': W, 'Y':Y}

    
    def estimate_W(self, Y, verbose=False):
        # Note: Since the TweedieRegressor's default penalty is L2, you just need to set alpha (which is equivalent to lambda in your case)
        with torch.no_grad():
            VA = self.V * torch.sqrt(torch.exp(self.a.unsqueeze(0))).detach().numpy()
        results = []
        Y = Y.numpy()
        for i in range(Y.shape[0]):
            if verbose:
                print(f'Computing latent variable : {i} / {Y.shape[0]}')
            self.poissonSolver.fit(VA, Y[i])
            results.append(self.poissonSolver.coef_)

        W = torch.Tensor(np.vstack(results))

        return W

    def train_encoder(self, sample_size, optimizer, criterion, X=None, Y=None, Z=None, num_epochs=100, validation_data=None, verbose=True, sample=False, **kwargs):
        """
        Train a PyTorch model and return the model and the history of losses.

        :param x: The input data for model
        :param y: The labels for input data
        :param z: additional input data
        :param u: additional input data
        :param model: PyTorch model to train
        :param optimizer: Optimizer to use in training
        :param criterion: Loss function to use in training
        :param num_epochs: Number of training epochs. Default is 100.
        :param validation_data: Tuple of validation data (x_val, y_val, z_val, u_val). Default is None.
        :return: Tuple of trained model and history of losses.
        """
        # Switch model to training mode
        self.train()
        
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            if sample:
                with torch.no_grad():
                    data = self.sample(sample_size, X)
                    Y = data['Y']
                    Z = data['W']

            optimizer.zero_grad()

            # Forward pass
            outputs_z = self.encoder(X, Y)

            loss = criterion(outputs_z, Z)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if validation_data is not None:
                self.eval() # switch model to evaluation mode
                with torch.no_grad():
                    z_val, y_val  = validation_data.values()
                    val_z = self.encoder(X, y_val)
                    val_loss =  criterion(val_z, z_val)
                    val_losses.append(val_loss.item())
                self.train() # switch model back to training mode

            if verbose and (epoch % 10) == 0:
                print(f'Epoch: {epoch}, Training Loss: {loss.item()}, Val loss = {None if validation_data is None else val_loss}')

        return self, train_losses, val_losses


    def train_model(self, X, data, A_init = None, num_epochs=1000, transform=False, verbose=False, avg_last=50, lr=(1., 0.01), gamma=(.95,.95), clip_value=0.05, encoder_epochs=10):
        if A_init is not None:
            self.a = torch.nn.Parameter(A_init)
        
        sample_size = data['Y'].shape[0]

        criterion =  MELoss()
        # Define optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr[0])

        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma[0])


        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr[1])
        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=gamma[1])
        encoder_criterion = nn.MSELoss()


        # Store the value of model at each epoch
        a_values = []
        b_values = []

        # Training loop
        self.train()
        # We start by training the encoder

        for epoch in range(num_epochs):
            _, encoder_loss, _ = self.train_encoder(sample_size, X=X, optimizer=encoder_optimizer, criterion=encoder_criterion, num_epochs=encoder_epochs, validation_data=None, sample=True, verbose=False)

            optimizer.zero_grad()
            
            with torch.no_grad():
                # # simulate
                data_sim = self.sample(sample_size, X=X)
                # # train encoder

                # compute W and W_sim
                self.encoder.eval()
                W = self.encoder(X, data['Y'])
                W_sim = self.encoder(X, data_sim['Y'])

                # W = model.estimate_W(data['Y'])
                # W_sim = model.estimate_W(data_sim['Y'])
            
            linpar, _ = self(W, X)
            linpar_sim, _ = self(W_sim, X)

            loss = criterion(data['Y'], linpar, data_sim['Y'], linpar_sim, transform=transform)
            
            loss.backward()
            clip_grad_value_(self.parameters(), clip_value=clip_value)
            optimizer.step()
            

            # Update the schedulers
            scheduler.step()
            encoder_scheduler.step()
            
            # Store the value of model
            a = self.a.clone().detach().cpu().numpy()
            a_values.append(a)
            b = self.B.clone().detach().cpu().numpy().flatten()
            b_values.append(b)

            if verbose: 
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, model.a[:5]: {a[:5]}, Encoder loss: {encoder_loss[-1]}')
            
        aa = np.vstack(a_values)
        bb = np.vstack(b_values)

        A_hist = aa[np.linspace(0, aa.shape[0]-1, min(aa.shape[0], 100), dtype=int)]
        A_last = aa[-avg_last:]
        A_last_avg = np.mean(aa[-avg_last:], axis=0)

        B_hist = bb[np.linspace(0, bb.shape[0]-1, min(bb.shape[0], 100), dtype=int)]
        B_last = bb[-avg_last:]
        B_last_avg = np.mean(bb[-avg_last:], axis=0)

        results = {
            'A' : self.a.detach().cpu().numpy(),
            'A_hist' : A_hist,
            'A_last': A_last,
            'A_last_avg' : A_last_avg,
            'B' : self.B.detach().cpu().numpy(),
            'B_hist': B_hist,
            'B_last': B_last,
            'B_last_avg': B_last_avg
        }
        return results


class Encoder(nn.Module):
    def __init__(self, num_features, num_covariates, num_latents, num_hidden):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features + num_covariates, num_hidden),
            # nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            # nn.Dropout(0.8),
            nn.Tanh(),
            nn.Linear(num_hidden, num_latents))
    
    def forward(self, X, Y):
        # contaenate X and Y 
        XY = torch.cat((X, Y), dim=1)
        return self.fc(XY)


class MELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, linpar, ys, linpars, mask=None, transform=True):
        """Computes the loss. hat is recontructed y, ys is simulated"""

        if transform:
            with torch.no_grad():
                y_transformed =  torch.log(y.clone() + 1) 
                ys_transformed = torch.log(ys.clone() + 1)

            loss = -torch.sum(y_transformed* linpar - ys_transformed * linpars) / y.shape[0]

        else:
             loss = -torch.sum(y * linpar - ys * linpars) / y.shape[0] 
        return loss


def save_model_fit(sim_name, sim_iter, sim_path="", model=None, fit=None):
    """
    Save both the model and fit data to specified paths.
    
    Parameters:
    - sim_name (str): Base name of the simulation.
    - sim_iter (int): Iteration or version of the simulation.
    - sim_path (str, optional): Base directory path for saving. Defaults to the current directory.
    - model (torch.nn.Module, optional): PyTorch model to be saved. If None, no model will be saved.
    - fit (dict, optional): Dictionary containing fit data to be saved. If None, no fit data will be saved.
    """
    # Define folders
    sim_path = Path(sim_path) / sim_name
    sim_name = f'{sim_name}_{sim_iter}'
    model_dir = sim_path / "model"
    fit_dir = sim_path / "fit"

    # Create folders
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    if model is not None:
        path_model =  model_dir / (sim_name + ".pth")
        torch.save(model, path_model)

    if fit is not None:
        path_fit = fit_dir / (sim_name + ".npz")
        np.savez(path_fit, **fit)

def load_model_fit(sim_name, sim_iter, sim_path="", load_model=True, load_fit=True):
    """
    Load both the model and fit data from specified paths.
    
    Parameters:
    - sim_name (str): Base name of the simulation.
    - sim_iter (int): Iteration or version of the simulation.
    - sim_path (str, optional): Base directory path for loading. Defaults to the current directory.
    - load_model (bool, optional): Whether to load the model. If False, the returned model will be None.
    - load_fit (bool, optional): Whether to load the fit data. If False, the returned fit data will be None.
    
    Returns:
    - model (torch.nn.Module or None): Loaded PyTorch model or None if `load_model` is False.
    - fit (dict or None): Dictionary containing loaded fit data or None if `load_fit` is False.
    """

    # Define folders
    sim_path = Path(sim_path) / sim_name
    sim_name = f'{sim_name}_{sim_iter}'
    model_dir = sim_path / "model"
    fit_dir = sim_path / "fit"

    if load_model:
        path_model =  model_dir / (sim_name + ".pth")
        model = torch.load(path_model)
    else:
        model = None
    
    if load_fit:
        path_fit = fit_dir / (sim_name + ".npz") 
        fit_data = np.load(path_fit)
        fit = {key : fit_data[key] for key in fit_data}
    else:
        fit = None
    
    return model, fit


# Some other helper functions

def gen_Sigma_AR1(num_features, rho=0.8):
    # AR1 parameters

    # Generate AR1 covariance matrix
    cov_matrix = rho ** np.abs(np.subtract.outer(np.arange(num_features), np.arange(num_features)))

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Calculate V and A
    V = eigenvectors
    a = eigenvalues


    return cov_matrix, V, a

def compute_Sigma(V, a):
    S = V @ np.diag(a) @ V.T
    return S

def gen_X(n, num_covariates, intercept=True):
    X = torch.randn((n, num_covariates))
    if intercept and num_covariates > 0:
        X[:,0] = 1
    return X



