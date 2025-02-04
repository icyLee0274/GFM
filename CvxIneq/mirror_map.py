import torch
from torch import Tensor


def unit_ball_mirror_map(xs: Tensor, check: bool = True) -> Tensor:
    xx = torch.einsum("ik, ik -> i", xs, xs)
    if check: assert torch.all(xx < 1)
    return xs / (1 - xx.view(-1, 1))


def unit_ball_dual_map(ys: Tensor) -> Tensor:
    yy = torch.einsum("ik, ik -> i", ys, ys)
    return ys / (.5 + torch.sqrt(.25 + yy.view(-1, 1)))
