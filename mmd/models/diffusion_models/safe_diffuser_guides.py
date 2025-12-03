import torch
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
from mmd.models.diffusion_models.sample_functions import (
    extract, 
    apply_hard_conditioning, 
    guide_gradient_steps
)

class SafeDiffuserWrapper:
    def __init__(self, norm_mins, norm_maxs, obstacles_config, method='invariance_relax_cf'):
        pass