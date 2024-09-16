    """
    Methods used to simulate data and gllvms
    """

    import torch
    import torch.nn as nn
    import numpy as np

    
    def simulate_linpar(p):
        torch.randn(p)