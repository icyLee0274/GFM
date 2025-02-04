import torch
from torch import tensor, Tensor

from .example import Example
from misc import *


class Polytope2D(Example):

    def __init__(self, args):
        super().__init__(args)

    def init_domain(self):
        self.domain = LinearConstraint(
            tensor([
                [2., 1.],
                [-2., 1.],
                [1., -1.],
                [-1., -1.],
            ], device=self.device),
            tensor([2., 2., 1., 1.], device=self.device),
        )

    def init_training(self):
        if self.gen_sample:
            pass

    def gen0(self) -> float:
        pass
