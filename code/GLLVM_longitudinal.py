import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR # scheduler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class LongitudinalGLLVM(nn.Module):
    def __init__(self, num_var, num_latent, num_covar, num_period, response_types, intercept=True):
        super().__init__()
        self.setting = {
            "p":num_var, 
            "q":num_latent, 
            "k":num_covar, 
            "T":num_period,
            'response_types': response_types,
            'response_link': 
                {
                'binary' : lambda x: torch.logit(x),
                'ordinal': lambda x: torch.logit(x),
                'counts': lambda x: torch.log(x)
                },
            'response_linkinv':
                {
                'binary': lambda x: 1/(1+torch.exp(-x)),
                'ordinal': lambda x: 1/(1+torch.exp(-x)),
                'counts': lambda x: torch.exp(x)
                },
            'response_transform':
                {
                'binary' : lambda x: 2*x - 1,
                'ordinal': lambda x: 2*x - 1,
                'counts': lambda x: torch.log(x+1)
                },
            "intercept": intercept
        }

        self.encoder = Encoder(self.setting)
        self.decoder = Decoder(self.setting)
        self.sample = Sample(self.decoder, self.setting)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=.1)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=.95)

    
    def plot_cov(self, what="linpar", x=None):
        data_sample = self.sample(x=x)
        data =  data_sample[what].detach().view(n, -1).cpu().numpy()
        cov_matrix = np.cov(data, rowvar=False)

        plt.figure(figsize=(10,10))
        sns.heatmap(cov_matrix, annot=False, fmt='g')
        plt.show()
    
    def set_learning_rates(self, lr_model = None, lr_encoder=None):
        if lr_model is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_model
        
        if lr_encoder is not None:
            for param_group in self.encoder.optimizer.param_groups:
                param_group['lr'] = lr_encoder
        
    def transform_responses(self, y):
        y_transformed = torch.zeros_like(y)
        with torch.no_grad():
            for response_type, response_id in self.setting['response_types'].items():
                y_transformed[:,:,response_id] = self.setting['response_transform'][response_type](y[:,:,response_id])
        return y_transformed

    
    def fit(self, x, y, mask, epochs, lr_model=None, lr_encoder=None, phi_lb = -1, phi_ub = 1, varu_lb = 0.1, varu_ub=2):
        
        device = self.decoder.wz.device

        self.set_learning_rates(lr_model, lr_encoder)

        y_masked = y.clone()
        y_masked[mask] = 0.5
        criterion = MELoss(setting=self.setting)
        mseLoss = MSELoss(setting=self.setting)
        # create tensor dataset
        losses = []
        learning_rates = []
        learning_rates_encoder = []
        loading_values = []
        intercept_values = []
        coef_values = []


        for epoch in range(1, epochs +1):
            self.optimizer.zero_grad()

            # impute and sample
            with torch.no_grad():
                # data_true["y"] = model.impute(data_true["x"], data_true["y"], mask=mask["y"], nsteps=5)
                # simulate data from the current parameter values, Unconditionally (!)
                data_sim = self.sample(x=x)
                y_sim_masked = data_sim["y"].clone()

                # # TODO: Check imputation: this is waaaay better
                y_masked[mask] = 0.5
                y_sim_masked[mask] = 0.5

                # y_masked = self.mean_impute(y=y_masked, mask=mask)
                # y_sim_masked = self.mean_impute(y=y_sim_masked, mask=mask)

                # y_masked = self.impute(x, y_masked, mask, nsteps=1)
                # y_sim_masked = self.impute(x, y_sim_masked, mask, nsteps=1)

                
                # assert torch.eq(data_sim["x"], x).all().item()

                # compute the imputing values without the gradients
                zhat_sample, uhat_sample = self.encoder(x, y_masked, transform_response=True)
                zhat_sim, uhat_sim = self.encoder(data_sim["x"], y_sim_masked, transform_response=True)


            # train the encoder on the simulated data
            encoder_loss = self.encoder.fit(data_sim["x"], data_sim["y"], data_sim["z"], data_sim["u"],  epochs= 20) 
            
            # compute the decoded value
            linpar_sample, mean_sample = self.decoder(x, zhat_sample, uhat_sample)
            linpar_sim, mean_sim = self.decoder(data_sim["x"], zhat_sim, uhat_sim)

            # # impute values
            # data_sample["y"] = model.impute(data_sample["x"], data_sample["y"], mask, nsteps=2)
            # data_sim["y"] = model.impute(data_sim["x"], data_sim["y"], mask, nsteps=2)

            # model.decoder.wz.grad = mygrad
            loss = criterion(y_masked, linpar_sample,  y_sim_masked, linpar_sim, mask=mask)
            loss.backward()

            # Update nuisance parameters
            with torch.no_grad():
                self.sample.var_u.data = .8* self.sample.var_u.data + .2 * torch.var(uhat_sample) * (self.sample.var_u.data / torch.var(uhat_sim))
                self.sample.phi.data = .8 * self.sample.phi.data + .2 * self.compute_autocorr(zhat_sample) * self.sample.phi.data / self.compute_autocorr(zhat_sim)

                self.sample.phi.data = torch.clamp(self.sample.phi.data, phi_lb, phi_ub)
                self.sample.var_u.data = torch.clamp(self.sample.var_u.data, varu_lb, varu_ub)

                print(f'var_u: {self.sample.var_u}, phi: {self.sample.phi}')
            


            with torch.no_grad():   
                fit = mseLoss(y_masked, mean_sample, mask)

            self.optimizer.step()
            self.scheduler.step()

            losses.append(fit.item())
            learning_rates.append(self.optimizer.param_groups[0]['lr'])
            learning_rates_encoder.append(self.encoder.optimizer.param_groups[0]['lr'])
            loading_values.append(self.decoder.wz.clone().detach())
            coef_values.append(self.decoder.wx.clone().detach())
            intercept_values.append(self.decoder.bias.clone().detach())
            print(f"\nEpoch {epoch}/{epochs}, loss_fit = {fit.item():.2f}, encoder_loss = {encoder_loss.item():.2f}.")

        saved = {
            "losses": losses,
            "learning_rates": learning_rates,
            "learning_rates_encoder": learning_rates_encoder,
            "loading_values": loading_values,
            "coef_values": coef_values,
            "intercept_values": intercept_values
        }

        return saved


    def impute(self, x, y, mask, nsteps=10):
        """ Impute the missing values provided by the mask (True is missing) and return x imputed"""
        for _ in range(nsteps):
            z, u = self.encoder(x,y, transform_response = True)
            _, mean = self.decoder(x,z,u)
            y[mask] = mean[mask]
        return y
    
    def mean_impute(self, y, mask):
        var_mean = torch.mean(y, dim=2).unsqueeze(2)
        ymean = torch.ones_like(y) * var_mean
        y[mask] = ymean[mask]
        return y

    def compute_autocorr(self, z):
        # Compute the sample autocovariance and autocorrelation
        autocovariance = torch.mean((z[:, 1:] - torch.mean(z[:, 1:])) * (z[:, :-1] - torch.mean(z[:, :-1])), dim=1)
        autocorrelation = autocovariance / torch.var(z[:, :-1])

        # Estimate phi using the autocorrelation formula for AR(1)
        phi = torch.mean(autocorrelation)

        return phi

class Sample(nn.Module):
    def __init__(self, decoder, setting):
        super().__init__()
        self.decoder = decoder
        self.setting = setting
        self.phi = nn.Parameter(torch.ones(1) * .8)#.detach() # we do not update it using backprop, but still define as a nn.Parameter to easily change between devices using nn.Module.to()
        self.var_u = nn.Parameter(torch.ones(1) * 1.0)#.detach()
    
    def forward(self, x, n=None):
        device = self.phi.device
        with torch.no_grad():
            if x is None:
                Warning("x was set to None for sampling. X is usually fixed. Are you sure you want to sample x?")
                x = torch.randn((n, self.setting['T'], self.setting['k']-1)).to(device)
                # Add intercepts and time information
                time_data = torch.from_numpy(np.linspace(0,4,9)).float().expand(x.shape[0], -1).unsqueeze(2).to(device)
                x = torch.cat([x, time_data], dim=2)
            
            n = x.shape[0]

            u = torch.randn((n, 1, self.setting['p'])).to(device) * torch.sqrt(self.var_u)
            d = torch.randn((n, self.setting['T'], self.setting['q'])).to(device)
            z = self.AR(d)

            linpar, mean = self.decoder(x, z, u) # decoder gives the expectation

            y = self.sample_response(mean)

            return {"x":x, "y":y, "z":z, "u":u, "linpar":linpar, "mean":mean}

    def sample_response(self, mean):
        device = self.phi.device
        y = torch.zeros_like(mean).to(device)
        for response_type, response_id in self.setting['response_types'].items():
            if response_type == "binary":
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
            elif response_type == "counts":
                y[:,:,response_id] = torch.poisson(mean[:,:,response_id])
        return y

    def AR(self, d):
        z = d.clone()

        for t in range(1, z.shape[1]):
            z[:,t] = z[:,t] + z[:, t-1].clone() * self.phi  # we need to clone else the gradient wants to pass through it
        return z
                
    
class Decoder(nn.Module):
    # Yields the expectation
    def __init__(self, setting):
        super().__init__()
        self.setting = setting
        # decoder part (our parameters of interest)
        self.wz = nn.Parameter(torch.randn((self.setting['q'], self.setting['p'])) * 1.2)
        self.wx = nn.Parameter(torch.randn((1, self.setting['k'], self.setting['p']))* .2) # Measurement invariance!
        self.bias = nn.Parameter(torch.zeros((1, 1, self.setting['p']))* .2) # Measurement invariance!

    # decoding (computing the conditional mean)
    def forward(self, x, z, u):

        xwx = (x.unsqueeze(2) @ self.wx).squeeze() # see section "details of tensorproducts"
        zwz = (z.unsqueeze(2) @ self.wz).squeeze()
        # for the ordinal variables:

        if self.setting['intercept']:
            linpar = self.bias + xwx + zwz + u 
        else:
            linpar = xwx + zwz + u 


        # Apply the inverse link to get the conditional expectation
        mean  = torch.zeros_like(linpar)
        for response_type, response_id in self.setting['response_types'].items():
            mean[:,:,response_id] = self.setting['response_linkinv'][response_type](linpar[:,:,response_id])
        # Transform the 
        return linpar, mean


class Encoder(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting=setting
        # encoder part
        # input dimension is (p+k) (responses + covariates)
        # output dimension is q+p (one latent )
        input_size = self.setting['p'] + self.setting['k']
        hidden_size = (self.setting['p'] + self.setting['q']) * 5
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
        )
        # fully connected layers for Z and U
        self.fc_Z = nn.Sequential(
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = self.setting['q']),
        )

        self.fc_U = nn.Sequential(
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features = self.setting['p'])
        )

        self.optimizer =  torch.optim.Adam(self.parameters(), lr=.01)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=.95)
        self.loss = nn.MSELoss()

    
    def forward(self, x, y, transform_response = False):
        # Initialize hidden state

        # pass the input through the RNN
        if transform_response:
            y_transformed = torch.zeros_like(y)
            with torch.no_grad():
                for response_type, response_id in self.setting['response_types'].items():
                    y_transformed[:,:,response_id] = self.setting['response_transform'][response_type](y[:,:,response_id])
        else:
            y_transformed = y

        xy = torch.cat([x, y_transformed], dim=2)
        rnn_out, _ = self.rnn(xy)
        out = self.fc(rnn_out)
        z_pred = self.fc_Z(out)
        u_pred = self.fc_U(out[:, -1, :]).unsqueeze(1)
        return z_pred, u_pred
    
    def fit(self, x, y, z, u, epochs = 100, verbose = False):
        y = y.clone()
        with torch.no_grad():
            for response_type, response_id in self.setting['response_types'].items():
                y[:,:,response_id] = self.setting['response_transform'][response_type](y[:,:,response_id])
        # Fit the encoder
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            z_pred, u_pred = self(x, y, transform_response = False)

            loss = self.loss(z_pred, z) + self.loss(u_pred, u)

            if verbose:
                print(f"\nEpoch {epoch}/{epochs}, loss={loss}")
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss
    
    def plot(self, x, y, z, u):
        with torch.no_grad():
            z_pred, u_pred = self(x, y, transform_response = True)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

        ax1.scatter(z, z_pred)
        ax1.set_xlabel('Z True')
        ax1.set_ylabel('Z Pred')
        ax1.set_title('Encoded values of Z')
        ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls="--", color="red")

        ax2.scatter(u, u_pred)
        ax2.set_xlabel('U True')
        ax2.set_ylabel('U Pred')
        ax2.set_title('Encoded values of U')
        ax2.plot(ax2.get_xlim(), ax2.get_ylim(), ls="--", color="red")


        plt.show()
        

class MSELoss(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

    def forward(self, y, mean, mask=None):
        """Computes the fit loss."""
        if mask is not None:
            return torch.sum(torch.pow(y - mean, 2) * ~mask) / torch.sum(~mask)
        else:
            return torch.mean(torch.pow(y - mean, 2))

class MELoss(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting
    
    def forward(self, y, linpar, ys, linpars, mask=None):
        """Computes the loss. hat is recontructed y, ys is simulated"""
        y_transformed = torch.zeros_like(y)
        ys_transformed = torch.zeros_like(ys)
        with torch.no_grad():
            for response_type, response_id in self.setting['response_types'].items():
                y_transformed[:,:,response_id] = self.setting['response_transform'][response_type](y[:,:,response_id])
                ys_transformed[:,:,response_id] = self.setting['response_transform'][response_type](ys[:,:,response_id])


        if mask is not None:
            return -torch.sum(y_transformed* linpar * ~mask - ys_transformed * linpars* ~mask)/y.shape[0]
        else:
            return -torch.sum(y_transformed* linpar - ys_transformed * linpars) / y.shape[0]
        # return torch.mean(torch.pow(y-linpar, 2) - torch.pow(ys - linpars, 2))
        # loss = torch.mean(y.T @ linpar - ys.T @linpars)/y.shape[0]
        # return loss
