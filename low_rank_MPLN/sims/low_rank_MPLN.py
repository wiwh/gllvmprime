import numpy as np
import torch
from torch.nn.utils import clip_grad_value_
from torch import nn
from sklearn.linear_model import PoissonRegressor

def gen_Sigma(num_features, rho=0.8):
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

def gen_X(n, num_covariates, intercept=True):
    X = torch.randn((n, num_covariates))
    if intercept and num_covariates > 0:
        X[:,0] = 0
    return X

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
    
    def forward(self, W, X=None, invlink=torch.exp):
        if X is None:
            X = torch.zeros((W.shape[0], 0))
        # Get the linpar (Z above) from the eigendecomposition
        VA = self.V * torch.sqrt(torch.exp(self.a.unsqueeze(0))) # TODO: a is in fact log_a
        linpar = X @ self.B.T + W @ VA.T
        condmean = linpar if invlink is None else invlink(linpar)
        return linpar, condmean

    def sample(self, n, X=None, poisson=True):
        with torch.no_grad():
            W = torch.randn((n, self.num_latents))

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

def train_encoder(X, Y, Z, model, optimizer, criterion, num_epochs=100,validation_data=None, verbose=True, random=False, sample=False, **kwargs):
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
    model.train()
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if sample:
            with torch.no_grad():
                data = model.sample(Y.shape[0], X)
                Y = data['Y']
                Z = data['W']

        optimizer.zero_grad()

        # Forward pass
        outputs_z = model.encoder(X, Y)

        loss = criterion(outputs_z, Z)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if validation_data is not None:
            model.eval() # switch model to evaluation mode
            with torch.no_grad():
                z_val, y_val  = validation_data.values()
                val_z = model.encoder(X, y_val)
                val_loss =  criterion(val_z, z_val)
                val_losses.append(val_loss.item())
            model.train() # switch model back to training mode

        if verbose and (epoch % 10) == 0:
            print(f'Epoch: {epoch}, Training Loss: {loss.item()}, Val loss = {None if validation_data is None else val_loss}')

    return model, train_losses, val_losses

def train_model(model, data, A_init = None, num_epochs=1000, transform=False, verbose=False):
    if A_init is not None:
        model.a = torch.nn.Parameter(model_true.a.clone()) # Start at the solution for simulations.... a bit cheating but not too much

    criterion =  MELoss()
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)


    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.01)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.99)
    encoder_criterion = nn.MSELoss()


    # Store the value of model.a at each epoch
    a_values = []

    # Training loop

    for epoch in range(num_epochs):
        _, encoder_loss, _ = train_encoder(X, data['Y'], data['W'], model=model, optimizer=encoder_optimizer, criterion=encoder_criterion, num_epochs=5, validation_data=None, sample=True, verbose=False)

        optimizer.zero_grad()
        
        with torch.no_grad():
            # # simulate
            data_sim = model.sample(n)
            # # train encoder

            # compute W and W_sim
            W = model.encoder(X, data['Y'])
            W_sim = model.encoder(X, data_sim['Y'])

            # W = model.estimate_W(data['Y'])
            # W_sim = model.estimate_W(data_sim['Y'])
        
        linpar, _ = model(W)
        linpar_sim, _ = model(W_sim)

        loss = criterion(data['Y'], linpar, data_sim['Y'], linpar_sim, transform=transform)
        
        loss.backward()
        clip_grad_value_(model.parameters(), clip_value=0.2)
        optimizer.step()
        

        # Update the schedulers
        scheduler.step()
        encoder_scheduler.step()
        
        # Store the value of model.a
        a = model.a.clone().detach().cpu().numpy()
        a_values.append(a)

        if verbose: 
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, model.a[:5]: {a[:5]}, Encoder loss: {encoder_loss[-1]}')
        
    aa = np.vstack(a_values)

    A_hist = aa[np.linspace(0, aa.shape[0]-1, max(aa.shape[0], 100), dtype=int)]
    A_last = aa[-100:]

    results = {
        'A' : model.a.detach().numpy(),
        'A_hist' : A_hist,
        'A_last': A_last
    }
    return results

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

