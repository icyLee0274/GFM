from typing import Tuple

import torch
from torch import Tensor


def _solve_intersection(b: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    bb = torch.einsum("ik,ik->i", b, b)  # b^Tb, shape n*1
    bv = torch.einsum("ik,ik->i", b, v)  # b^Tv, shape n*1
    vv = torch.einsum("ik,ik->i", v, v)  # v^Tv, shape n*1

    # Solve intersection  || b + t v || = 1
    t = (-bv + torch.sqrt(bv * bv + vv - vv * bb)) / vv
    t = torch.nan_to_num(t)
    t = torch.minimum(t, torch.ones_like(t))
    t = t.view(-1, 1)

    return bb, bv, vv, t


def ball_projection(b: Tensor, v: Tensor) -> Tensor:
    d = b + v  # step destination
    n = torch.linalg.vector_norm(d, dim=1)  # norm of d
    n[n > 1] += 1e-6
    n[n < 1] = 1  # no projection
    return d / n.view(-1, 1)


def ball_reflect(b: Tensor, v: Tensor) -> Tensor:
    """
    Reflection functino on unit sphere.

    :param b: 2D Tensor, start point of iteration.
    :param v: 2D Tensor, step vector, not assumed to be unit vector.
    :return: 2D Tensor, end point, reflect if needed.
    """
    bb, bv, vv, t = _solve_intersection(b, v)

    n = b + t * v  # normal vector
    # If reflection is required, w is the remaining step after intersection point,
    # project w onto normal vector by w_n = w^Tnn
    w = (1 - t) * (bv.view(-1, 1) + t * vv.view(-1, 1)) * n

    return b + v - 2 * w


def ball_reflect_velocity(b: Tensor, v: Tensor, velocity_field: Tensor) -> Tensor:
    """
    Reflected velocity functino on unit sphere.

    :param b: 2D Tensor, start point of iteration.
    :param v: 2D Tensor, step vector, not assumed to be unit vector.
    :param velocity_field: 2D Tensor
    :return: Negative velocity if reflection is required
    """
    bb, bv, vv, t = _solve_intersection(b, v)

    cond = torch.floor(t)  # 0 if need reflection, 1 otherwise
    cond = -1 + 2 * cond

    return cond.view(-1, 1) * velocity_field
