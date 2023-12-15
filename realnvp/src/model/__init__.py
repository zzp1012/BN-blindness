import torch
import torch.nn as nn
import torch.distributions as distributions

# import internal libs
from model.realnvp import RealNVP_BN
from model.realnvp_ln import RealNVP_LN
from data import DataInfo
from utils import get_logger

class Hyperparameters():
    def __init__(self, 
                 base_dim: int, 
                 res_blocks: int, 
                 bottleneck: int, 
                 skip: int, 
                 weight_norm: int, 
                 coupling_bn: int, 
                 affine: int):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine


def build_model(device: torch.device,
                data_info: DataInfo,
                base_dim: int,
                res_blocks: int,
                bottleneck: int,
                skip: int,
                weight_norm: int,
                coupling_bn: int,
                affine: int,
                bn_type: str) -> nn.Module:
    """prepare the realNVP model.

    Args:
        device: device to use.
        data_info: data information.
        base_dim: features in residual blocks of first few layers.
        res_blocks: number of residual blocks to use.
        bottleneck: True if use bottleneck, False otherwise.
        skip: True if use skip architecture, False otherwise.
        weight_norm: True if apply weight normalization, False otherwise.
        coupling_bn: True if batchnorm coupling layer output, False otherwise.
        affine: True if use affine coupling, False if use additive coupling.
        bn_type: type of batchnorm.
    
    Returns:
        model: realNVP model.
    """
    logger = get_logger(__name__)
    
    hps = Hyperparameters(
        base_dim = base_dim, 
        res_blocks = res_blocks, 
        bottleneck = bottleneck, 
        skip = skip, 
        weight_norm = weight_norm, 
        coupling_bn = coupling_bn, 
        affine = affine)
    logger.info(f"Hyperparameters: {hps.__dict__}")

    prior = distributions.Normal(   # isotropic standard normal distribution
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    if bn_type == "bn":
        flow = RealNVP_BN(device=device, datainfo=data_info, prior=prior, hps=hps).to(device)
    elif bn_type == "ln":
        flow = RealNVP_LN(device=device, datainfo=data_info, prior=prior, hps=hps).to(device)
    else:
        raise ValueError(f"Invalid bn_type: {bn_type}")
    return flow