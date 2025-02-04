import torch
from torch import Tensor
from misc import ConstrainedSet

__all__ = ["GaugeMap"]


class GaugeMap:

    def __init__(self, constraint: ConstrainedSet, ref_point: Tensor):
        """
        Gauge mapping between a (compact) convex constrained set and the unit ball.

        :param constraint: Constrained set.
        :param ref_point: Interior point of the constrained set,
            its feasibility is not checked, please make sure it is strictly feasible.
        """
        self.constraint = constraint
        self.x0 = ref_point.view(1, -1)

    def to_disk(self, xs: Tensor, tol: float = 1e-6, thresh: float = 1e8, device=torch.get_default_device()) -> Tensor:
        vs = xs - self.x0
        ds = torch.linalg.vector_norm(vs, dim=-1)

        ii = ds >= tol

        vs = vs[ii]
        zs = torch.zeros_like(xs)
        ps = self.constraint.eval_intersection_v(self.x0.expand(vs.shape[0], -1), vs, tol, thresh, device)
        zs[ii, :] = vs / (ps * ds[ii]).view(-1, 1)

        return zs

    def from_disk(self, zs: Tensor, tol: float = 1e-6, thresh: float = 1e8,
                  device=torch.get_default_device()) -> Tensor:
        # z = x / ( |x| * phi(x) ) => |z| = 1 / phi(x)
        ds = torch.linalg.vector_norm(zs, dim=-1, keepdim=True).to(device)
        ii = torch.flatten(ds >= tol)

        xs = torch.zeros_like(zs, device=device)
        x0 = self.x0.to(device)
        xs[~ii] = x0

        vs = zs[ii] / ds[ii]
        ts = (self
              .constraint
              .eval_intersection_v(self.x0.expand(vs.shape[0], -1), vs, tol, thresh, device)
              .view(-1, 1))

        xs[ii] = vs * ts * ds[ii] + x0

        return xs
