import logging
from itertools import product
from math import sqrt

from cvxpy import NonPos
from omegaconf import DictConfig
import torch, numpy as np
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
import cvxpy as cp
import geoopt

import gfm
import examples
from gfm import ConstrainedSet

logger = logging.getLogger(__name__)


class Combinatorial(examples.GfmExampleBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
