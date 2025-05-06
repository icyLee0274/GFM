import torch, numpy
from torch import Tensor
from typing import Tuple

__all__ = ["ball_reflect", "cube_reflect", "PolytopeReflector", "QcReflector"]


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


def cube_reflect(o: Tensor, v: Tensor) -> Tensor:
    e = o + v
    e[e > 1] = 2 - e[e > 1]
    e[e < -1] = -2 - e[e < -1]
    return e


class PolytopeReflector:

    def __init__(self, At: torch.Tensor, b: torch.Tensor):
        self.b = b
        self.A = At.T
        self.At = At

    def __call__(self, os: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        A = self.A.to(os.device)  # K * d
        At = self.At.to(os.device)  # d * K
        b = self.b.to(os.device)
        ao = os @ At  # n * K
        av = vs @ At  # n * K
        ii = torch.isclose(av, torch.zeros_like(av))

        av[ii] = 1
        ts = (b - ao) / av  # n * K
        ts[ii] = torch.inf
        ts[ts < 0] = torch.inf
        ts, jj = torch.min(ts, dim=1)
        ts = torch.minimum(ts, torch.ones_like(ts))

        xs = os + ts.view(-1, 1) * vs  ## end point before reflection
        ns = A[jj]  ## n * d, normal vectors
        nv = torch.einsum("ik, ik -> i", ns, vs).view(-1, 1)
        nn = torch.einsum("ik, ik -> i", ns, ns).view(-1, 1)
        rs = vs - 2 * (nv / nn) * ns  ## reflected direction

        return xs + (1 - ts.view(-1, 1)) * rs


class QcReflector:

    def __init__(self, Hs: Tensor, Gs: Tensor, fs: Tensor):
        """
        Qc constraints
            1/2 x^T H_i x + G_i^T x + f_i <= 0

        :param Hs:
        :param Gs:
        :param fs:
        """
        self.Hs = Hs.detach()
        self.Gs = Gs.detach()
        self.fs = fs.detach()
        self.n_constraints = Hs.shape[0]

    # def norm_vector(self, xs: Tensor)  -> Tensor :

    def __call__(self, os: Tensor, vs: Tensor) -> Tensor:
        t_all = torch.empty([os.shape[0], self.n_constraints], device=os.device)
        for i_con in range(self.n_constraints):
            vHv = torch.einsum("ni, ij, nj -> n", vs, self.Hs[i_con], vs)
            oHo = torch.einsum("ni, ij, nj -> n", os, self.Hs[i_con], os)
            oHv = torch.einsum("ni, ij, nj -> n", os, self.Hs[i_con], vs)
            Gv = torch.matmul(vs, self.Gs[i_con])
            Go = torch.matmul(os, self.Gs[i_con])

            a = vHv
            b = Gv + 2 * oHv
            c = oHo + Go + self.fs[i_con]
            delta = b * b - 4 * a * c

            ii = torch.logical_or(delta < 0, torch.abs(a) < 1e-8)
            delta[ii] = 0
            a[ii] = 1

            ts = (-b + torch.sqrt(delta)) / 2 / a
            ts[ii] = torch.inf
            t_all[:, i_con] = ts
        # t_all: N*k, corresponding step length 't' for each constraint
        ts, ii = torch.min(t_all, dim=1)  # ts: N, step length after consider all constraints;
        #                                 # ii: N, which constraint is hit
        ts = torch.minimum(ts, torch.ones_like(ts))  # Limit step length to 1
        ds = ts.view(-1, 1) * vs + os  # destination (before reflection)
        # This loop might be vectorized -TODO
        for i in range(os.shape[0]):
            if ts[i] < 1:  # if hit one constraint => need reflection
                H = self.Hs[ii[i]]  # corresponding constraint
                G = self.Gs[ii[i]]

                n = - H @ ds[i] - G  # normal vector at 'ds[i]'
                nv = n @ vs[i]
                nn = n @ n
                r = vs[i] - 2 * (nv / nn) * n
                ds[i] = ds[i] + (1 - ts[i]) * r
        return ds
