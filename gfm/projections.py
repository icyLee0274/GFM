import torch
import numpy as np
from quadprog import solve_qp
import cvxpy as cp
from torch import Tensor

__all__ = ["PolytopeProjector", "cube_project", "ball_project", "EllipsoidProjector"]


class EllipsoidProjector:

    def __init__(self, Qs: Tensor, ps: Tensor, bs: Tensor, G: Tensor | None = None, h: Tensor | None = None):
        """
        Quadratic constraints defined as
            1/2 x^T Q_i x + p^T x + b <= 0, for i = [1,K],
            G x <= h.

        :param Qs: Batched matrices of size K*d*d.
        :param ps: Batched vectors of size K*d.
        :param bs: Batched constants of size K.
        :param G: Matrix of size L*d.
        :param h: Vector of size L.
        """
        with torch.no_grad():
            self.Qs = Qs.cpu().numpy()
            self.ps = ps.cpu().numpy()
            self.bs = bs.cpu().numpy()
            self.G = G.cpu().numpy() if G is not None else None
            self.h = h.cpu().numpy() if h is not None else None
            self.dim = Qs.shape[1]
            self.n_constraints = Qs.shape[0]
            self.In = np.eye(self.dim)

    def do_projection(self, y: Tensor) -> Tensor:
        with torch.no_grad():
            x = cp.Variable(self.dim)
            yt = y.numpy().T
            objective = cp.Minimize(.5 * cp.QuadForm(x, self.In) - yt @ x)
            constraints = [.5 * cp.QuadForm(x, self.Qs[i]) + self.ps[i] @ x + self.bs[i] <= self.bs
                           for i in range(self.n_constraints)]
            if self.G is not None: constraints = constraints + [self.G @ x <= self.h]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            return torch.from_numpy(x.value).to(y.device)

    def __call__(self, os: Tensor, vs: Tensor) -> Tensor:
        ds = os + vs
        for i in range(ds.shape[0]):
            ds[i] = self.do_projection(ds[i])
        return ds


class PolytopeProjector:

    def __init__(self, At: Tensor, b: Tensor, box_lower: float | None = None, box_upper: float | None = None):
        with torch.no_grad():
            self.At = At
            self.b = b

            self.G = np.eye(At.shape[0])
            self.C = -At.numpy().astype(np.double)
            self.b_numpy = -b.numpy().flatten().astype(np.double)

            self.box_lower = box_lower
            self.box_upper = box_upper
            self.boxed = box_lower is not None or box_upper is not None

    def do_projection(self, y: Tensor) -> Tensor:
        with torch.no_grad():
            q = y.numpy().flatten().astype(np.double)
            # sol = cvxopt.solvers.qp(self.P, q, self.G, self.h)
            x, _, _, _, _, _ = solve_qp(self.G, q, self.C, self.b_numpy, meq=0, factorized=True)
            return torch.from_numpy(x.flatten()).to(y.device)

    def __call__(self, os: Tensor, vs: Tensor) -> Tensor:
        ds = os + vs
        At = self.At.to(ds.device)
        b = self.b.to(ds.device)
        fs = torch.any(ds @ At > b, dim=1)
        for i in range(ds.shape[0]):
            if fs[i]:
                ds[i] = self.do_projection(ds[i])

        if self.boxed:
            ds[ds > self.box_upper] = self.box_upper
            ds[ds < self.box_lower] = self.box_lower

        return ds


def cube_project(o: Tensor, v: Tensor) -> Tensor:
    e = o + v
    e[e > 1] = 1
    e[e < -1] = -1
    return e


def ball_project(b: Tensor, v: Tensor) -> Tensor:
    d = b + v  # step destination
    n = torch.linalg.vector_norm(d, dim=1)  # norm of d
    n[n > 1] += 1e-6
    n[n < 1] = 1  # no projection
    return d / n.view(-1, 1)
