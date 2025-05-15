from typing import Literal

import geoopt
import numpy as np
import torch
from torch import Tensor
from .constraints import ConstrainedSet

__all__ = ["GaugeMap"]


class GaugeMap:

    def __init__(
            self,
            constraint: ConstrainedSet,
            ref_point: Tensor,
            dest: Literal["cube", "ball"] = "cube",
            manifold: geoopt.Manifold | None = None,
    ):
        """
        Gauge mapping between a (compact) convex constrained set and the unit ball.

        :param constraint: Constrained set.
        :param ref_point: Interior point of the constrained set,
            its feasibility is not checked, please make sure it is strictly feasible.
        """
        self.constraint = constraint
        self.x0 = ref_point.view(1, -1)
        self.dest = dest
        self.manifold = manifold
        match dest:
            case "cube":
                self.p = torch.inf
            case "ball":
                self.p = 2

    def log_map(self, xs: Tensor, ys: Tensor) -> Tensor:
        return (self.manifold.logmap(xs, ys) if self.manifold is not None
                else ys - xs)

    def exp_map(self, xs: Tensor, ys: Tensor) -> Tensor:
        return (self.manifold.expmap(xs, ys) if self.manifold is not None
                else xs + ys)

    def boundary_dist(
            self,
            x0: Tensor,
            us: Tensor,
            tol,
            thresh,
            device,
            **kwargs,
    ) -> Tensor:
        ds = (
            self.constraint.eval_manifold_intersection_v(x0, us, self.manifold, tol, thresh, device, **kwargs)
            if self.manifold is not None else
            self.constraint.eval_intersection_v(x0, us, tol, thresh, device)
        )
        return ds.view(-1, 1)

    def to_disk(
            self,
            xs: Tensor,
            tol: float = 1e-6,
            thresh: float = 1e8,
            device=torch.get_default_device(),
            **kwargs,
    ) -> Tensor:
        """
        Maps points on manifold into (scaled) unit ball.

        If the manifold is provided, the unit ball is inside the tangent space of the reference point

        This method implements the inverse (geodesic) gauge map:

        .. math::
            \Phi^{-1}(x) &= \gamma_\mathcal{U}(x, x^\circ) v_x \\
                         &= r_x v_x / d_\mathcal{U}(x^\circ, v_x) \\
                         &= u_x / d_\mathcal{U}(x^\circ, v_x) \\

        Where
            - :math:`\mathcal{U}` is a bounded (geodesic) convex set (over a manifold).
            - :math:`\gamma_\mathcal{U}` is the gauge function.
            - :math:`u_x = r_x v_x = \log_{x^\circ}(x)`
            - :math:`d_\mathcal{U}` is the distance to boundary function

        :param xs: Points on manifold to map, assumed to be already projected onto the manifold.
        :param tol: Tolerance.
        :param thresh: Threshold.
        :param device: Torch device.
        :param kwargs: Arguments passing to: method:`GaugeMap.boundary_dist`, :see:method:`ConstrainedSet.eval_manifold_intersection_v`.
        :return: Tensor of mapped points inside the unit ball in the tangent space.
        """
        x0 = self.x0.expand(xs.shape[0], -1).to(xs)
        # tangent vectors at x0 to xs
        us = self.log_map(x0, xs)
        # distances from x0 to xs
        rs = torch.linalg.vector_norm(us, dim=-1, keepdim=True, ord=self.p).to(us)
        ii = torch.flatten(rs >= tol)
        # normalize the tangent vectors
        vs = us[ii] / rs[ii]
        zs = torch.zeros_like(xs)
        # point-to-boundary distance: d_\mathcal{U}(x^\circ, v_x)
        ds = self.boundary_dist(x0[ii], vs, tol, thresh, xs.device, **kwargs)
        zs[ii, :] = us / ds

        return zs

    def from_disk(
            self,
            zs: Tensor,
            tol: float = 1e-6,
            thresh: float = 1e8,
            device=torch.get_default_device(),
            **kwargs,
    ) -> Tensor:
        """
        Maps points, which are in the (scaled) unit ball (in the tangent space of the reference point),
        to the (geodesic) convex set (over the manifold).

        If the manifold is provided, the unit ball is inside the tangent space of the reference point,
        and the mapped points are on the manifold.

        This method implements the (geodesic) gauge map:
        .. math::
                \Phi(z) &= \exp_{x^\circ}(d_\mathcal{U}(x^\circ, z/||z||) \cdot z) \\
                        &= \exp_{x^\circ}(||z|| d_\mathcal{U}(x^\circ, z) \cdot z) \\

        where
                - :math:`d_\mathcal{U}` is the distance to boundary function.
                - :math:`\exp_{x^\circ}` is the exponential map.

        :param zs: Points in the unit ball in the tangent space, not checked.
        :param tol: Tolerance.
        :param thresh: Threshold.
        :param device: Torch device.
        :return: Points on the manifold.
        """
        # z = x / ( |x| * phi(x) ) => |z| = 1 / phi(x)

        x0 = self.x0.expand(zs.shape[0], -1).to(zs)

        # rs
        rs = torch.linalg.vector_norm(zs, dim=-1, keepdim=True, ord=self.p).to(zs)

        ii = torch.flatten(rs >= tol)
        xs = torch.zeros_like(zs)
        xs[~ii] = x0[~ii]

        ds = self.boundary_dist(x0[ii], zs[ii], tol, thresh, zs.device, **kwargs)
        xs[ii, :] = self.exp_map(x0[ii], ds * rs[ii] * zs[ii])

        return xs
