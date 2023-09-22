import low_rank_MPLN
import torch
import numpy as np

def sim31_gen_models(q, p, n):
    
    num_latents = q
    num_features = p

    num_covariates = 3 # We include 1 intercept, presumably the "B" in the original docuzment. The true values of the corresponding coefficient is 0 here.
    # Generate Sigma and obtain its eigendecomposition:
    Sigma, V, _ = low_rank_MPLN.gen_Sigma_AR1(num_features, rho=0.8)
    a = np.linspace(10, 5, num_latents)

    # Generate the true model and sample some data
    model_true = low_rank_MPLN.MPLN(num_latents, num_features, num_covariates, V[:,:num_latents], np.log(a[:num_latents]))
    X = sim31_gen_X(n)
    B = sim31_gen_B(num_features)
    model_true.B = torch.nn.Parameter(B)

    # Define the estimated model and train it
    model = low_rank_MPLN.MPLN(num_latents, num_features, num_covariates, V[:,:num_latents])
    model.load_state_dict(model_true.state_dict())

    return(model_true, model, X)

def sim31_gen_X(n):
    b1 = torch.ones((n,))
    b2 = torch.log(torch.poisson(torch.ones((n,))*10) + 1)
    b3 = torch.rand((n,)) < 0.4
    X = torch.stack([b1,b2,b3], dim=1)

    return (X)


def sim31_gen_B(num_features):
    # set the parameters of the model to that of sim_31
    b1_true = torch.ones((num_features,)) * 0.
    b2_true = torch.ones((num_features,)) * 0.1
    b3_true = torch.cat([
        torch.ones((2 ,)) * 0.8,
        torch.ones((18,)) * 0.2,
        torch.ones((30,)) * 0.1,
        torch.ones((50,)) * -0.1,
        torch.ones((num_features - 100),) * 0.
    ])
    b_true = torch.stack([b1_true, b2_true, b3_true], dim=1)

    return (b_true)

