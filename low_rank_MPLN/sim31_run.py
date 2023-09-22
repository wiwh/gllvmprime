import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_value_
from torch import nn
from sklearn.linear_model import PoissonRegressor
import matplotlib.pyplot as plt

sys.path.append(Path("low_rank_MPLN"))
import sim31
import low_rank_MPLN

# Simulation 1 setup
# --------------------
n = 10000  # number of observations
num_features = 400
num_latents = 20 # We suppose full rank for both the true and estimated model

sim_name = f'sim31_n{n}_p{num_features}_q{num_latents}'

print(f'Using cuda:{torch.cuda.is_available()}') 

for sim_iter in range(100):
    print(f'iteration: {sim_iter}')

    model_true, model, X = sim31.sim31_gen_models(n=n, q=num_latents, p=num_features)
    data = model_true.sample(n, X, poisson=True)

    #Training

    # Go to cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    model.to(device)
    model_true.to(device)
    data = {key:value.to(device) for key, value in data.items()}

    fit = model.train_model(X, data, verbose=True, num_epochs=1000, lr=(1., 0.01), gamma=(0.98, 0.98), clip_value=0.05, encoder_epochs=5)

    # Go back to cpu
    device = torch.device("cpu")
    X = X.to(device)
    model.to(device)
    model_true.to(device)
    data = {key:value.to(device) for key, value in data.items()}

    # save model
    low_rank_MPLN.save_model_fit(sim_name = sim_name, sim_iter = sim_iter, sim_path="low_rank_MPLN/sim31", model=model, fit=fit)
    low_rank_MPLN.save_model_fit(sim_name = sim_name + "_true", sim_iter = sim_iter, sim_path="low_rank_MPLN/sim31", model=model_true, fit=None)
