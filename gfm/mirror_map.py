import torch
from torch import Tensor


def unit_ball_mirror_map(xs: Tensor, check: bool = False) -> Tensor:
    xx = torch.einsum("ik, ik -> i", xs, xs)
    if check: assert torch.all(xx < 1)
    return xs / (1 - xx.view(-1, 1))


def unit_ball_dual_map(ys: Tensor) -> Tensor:
    yy = torch.einsum("ik, ik -> i", ys, ys)
    return ys / (.5 + torch.sqrt(.25 + yy.view(-1, 1)))


def unit_cube_mirror_map(xs: Tensor, check: bool = False) -> Tensor:
    return 1 / (1 - xs - 1e-12) - 1 / (1 + xs + 1e-12)


def unit_cube_dual_map(ys: Tensor) -> Tensor:
    return (torch.sqrt(1 + ys * ys) - 1) / ys
