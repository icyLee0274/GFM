from typing import Callable, Literal

import geoopt
import torch
from torch import tensor, Tensor

__all__ = [
    "ConstrainedSet",
    "Intersection",
    "LinearConstraint",
    "BallConstraint",
    "QuadraticConstraint",
    "ConeConstraint",
    "SemiDefiniteConstraint",
    "IneqConstraint",
]


class ConstrainedSet:
    """
    Base class for constrained sets.

    Feasibility of points can be checked by `check_feasibility()` or `check_feasibility_v()`.

        Methods:
            `check_feasibility`: check the feasibility of one point.
            `check_feasibility_v`: check the feasibility of some points.
    """

    def __init__(self, ):
        pass

    def check_feasibility(self, point: Tensor) -> bool:
        """
        Check the feasibility of one point.
        :param point: a one-dimensional Tensor representing a point.
        :return: if the point is feasible, aka, inside the constrained set.
        """
        raise NotImplementedError

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        res = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
        for i in range(points.shape[0]):
            res[i] = self.check_feasibility(points[i])
        return res

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        """
        Evaluate the intersection of boundary and ray with direction v from o.

        The returned value is the scale factor t of direction vector v, namely o + tv \in \partial S.

        :param o: The Starting point of the ray, assumed to be feasible, flattened to be an 1D Tensor.
        :param v: Direction vector of the ray, not necessarily to be unit, flattened to be an 1D Tensor.
        :param tol: Tolerance of bisection, default to 1e-6.
        :param thresh: Threshold for no intersection, if a scale is larger than the threshold, defaults to 1e8.
        :return: Scale t, inf if not intersection.
        """
        o = o.flatten()
        v = v.flatten()
        if self.check_feasibility(o + v):
            upper = 2
            while upper < thresh:
                if self.check_feasibility(o + upper * v):
                    upper *= 2
                else:
                    break
            if upper > thresh: return torch.inf
            lower = upper / 2
        else:
            lower = .5
            while lower > tol:
                if self.check_feasibility(o + lower * v):
                    break
                else:
                    lower /= 2
            if lower < tol: return 0
            upper = lower * 2

        mid = (lower + upper) / 2
        while upper - lower > tol:
            if self.check_feasibility(o + mid * v):
                lower = mid
            else:
                upper = mid
            mid = (lower + upper) / 2
        return mid

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device()) -> Tensor:
        ts = torch.ones(vs.shape[0], device=device)
        for i in range(vs.shape[0]):
            ts[i] = self.eval_intersection(os[i], vs[i], tol, thresh)
        return ts

    def eval_manifold_intersection_v(
            self,
            x0s: Tensor, vs: Tensor,
            manifold: geoopt.Manifold,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device(),
            **kwargs,
    ) -> Tensor:
        """
        Evaluate the intersection of a manifold boundary and rays defined by starting points and directions.
        Namely, compute the point-to-boundary distance d(x_0, v):

        .. math::
            x_0 + d(x_0, v) v \in \partial S

        This method computes the scale factors `t` for each ray such that the point `x0 + t * v` lies on the boundary
        of the manifold. The computation is performed using a combination of upper-bound estimation and bisection.

        For some manifolds, unlike Euclidean space, which is the :method:`eval_intersection_v` method,
        the ray `x0 + t * v` may intersect the boundary of the constrained set at multiple points.
        In such cases, this method returns the (possible) first intersection point.
        It is preferred to provide suitable `tol` and `thresh` values to ensure the method works correctly.
        If an upper bound on the scale of the tangent exists, provide it as `thresh`.
        Otherwise, choose smaller `step_size` and larger `n_slice` for correctness.

        :param x0s: Starting points of the rays, a 2D Tensor of shape (n, d).
        :param vs: Direction vectors of the rays, a 2D Tensor of shape (n, d).
        :param manifold: A geoopt.Manifold object representing the manifold.
        :param tol: Tolerance for the bisection method, default is 1e-6.
        :param thresh: Threshold for no intersection; scale factors larger than this are considered infinite, default is 1e8.
        :param device: Torch device to perform computations on, default is the current default device.
        :param kwargs: Additional parameters:
            - max_iter (int): Maximum number of iterations for bisection, default is 1e5.
            - step_size (float): Step size for finding the upper bound, default is 2.
            - n_slice (int): Number of slices for checking feasibility in the upper bound search, default is 100.
            - growth (str): Growth strategy for the step size, either "fixed" or "exponential", default is "fixed".
            - init (Tensor): Initial scale factors for the rays, default is a Tensor of ones.

        :return: A 1D Tensor of scale factors `t` for each ray. If no intersection is found, the value is `inf`.
        """

        max_iter = kwargs.get("max_iter", 1e5)
        step_size = kwargs.get("step_size", 2)
        n_slice = kwargs.get("n_slice", 100)
        growth = kwargs.get("growth", "exponential")  # "fixed" or "exponential"
        t0 = kwargs.get("init", None)
        if t0 is None: t0 = torch.ones(vs.shape[0], device=device)
        if not torch.is_tensor(t0): raise ValueError("`init` must be of type Tensor.")

        if x0s.dim() == 1: x0s = x0s.expand(vs.shape[0], -1)

        # Find the upper bound
        uppers = t0.clone()
        cont = torch.ones(vs.shape[0], dtype=torch.bool, device=device)

        while torch.any(cont):
            nxt = uppers + step_size if growth == "fixed" else uppers * step_size
            nxt[nxt > thresh] = thresh

            if n_slice > 0:
                for i in torch.nonzero(cont, as_tuple=False):
                    ts = torch.linspace(uppers[i].item(), nxt[i].item(), n_slice, device=device)
                    xs = manifold.expmap(x0s[i].expand(n_slice, -1), ts.view(-1, 1) * vs[i])
                    fs = self.check_feasibility_v(xs, device)
                    if torch.any(~fs):
                        uppers[i] = ts[~fs][0]
                        cont[i] = False
                uppers[cont] = nxt[cont]
            else:
                xs = manifold.expmap(x0s[cont], nxt[cont].view(-1, 1) * vs[cont])
                fs = self.check_feasibility_v(xs, device)
                uppers[cont] = nxt[cont]
                cont[cont] = fs

            cont = cont & (uppers < thresh)

        cont = torch.ones(vs.shape[0], dtype=torch.bool, device=device)
        lowers = uppers.clone()
        while torch.any(cont):
            lowers[cont] = lowers[cont] / step_size
            xs = manifold.expmap(x0s[cont], lowers[cont].view(-1, 1) * vs[cont])
            cont[cont] = ~self.check_feasibility_v(xs)
            cont[cont] = cont[cont] & (lowers > tol)

        # Bisection
        ts = t0
        cont = torch.ones(vs.shape[0], dtype=torch.bool, device=device)
        mask = torch.ones_like(cont)

        i = 0
        while torch.any(cont) and i < max_iter:
            i += 1

            ts = (lowers + uppers) / 2
            xs = manifold.expmap(x0s[cont], ts[cont].view(-1, 1) * vs[cont])
            fs = self.check_feasibility_v(xs, device)

            # feasible => ts are too small
            mask[cont] = fs
            lowers[mask] = ts[mask]
            # infeasible => ts are too large
            mask[cont] = ~fs
            uppers[mask] = ts[mask]
            cont = torch.flatten(uppers - lowers >= tol)

        return ts


class LinearConstraint(ConstrainedSet):

    def __init__(self, A: Tensor, b: Tensor):
        """
        Linear constrained set defined by

            A * x <= b

        :param A: A matrix of size K * d.
        :param b: Constant vector of length K.
        """
        super().__init__()
        self.At = A.transpose(0, 1)
        self.b = b.view(1, -1)

    def _eval_lhs(self, x: Tensor) -> Tensor:
        """
        Evaluate the lhs constraints, namely.

        .. math::
            A * x

        :param x: 2D Tensor of size n * p of points evaluated at.
        :return: A 2D Tensor os size n * K.
        """
        return torch.matmul(x, self.At.to(x))

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        return torch.all((points.mm(self.At.to(points)) <= self.b.to(points)), dim=1).to(device)

    def check_feasibility(self, point: Tensor) -> bool:
        return self.check_feasibility_v(point.view(1, -1), point.device)[0].item()

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        return self.eval_intersection_v(o.view(1, -1), v.view(1, -1), tol, thresh, o.device)[0].item()

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device()) -> Tensor:
        """
        Evaluate the intersection of boundary and ray with direction v from o.

            t = (b - a^T o) / a^T v
        """
        ao = self._eval_lhs(os)  # n * K
        av = self._eval_lhs(vs)  # n * K
        ii = torch.isclose(av, torch.zeros_like(av))  # n * K
        fs = torch.all(ao <= self.b.to(ao), dim=1).logical_not()  # (n,)

        av[ii] = 1
        ts = (self.b.to(ao) - ao) / av  # n * K
        ts[ts < 0] = torch.inf
        ts[ii] = torch.inf
        ts = torch.min(ts, dim=1)[0]
        ts[fs] = torch.inf
        return ts


class BallConstraint(ConstrainedSet):

    def __init__(self, loc: Tensor, scale: float = 1.0):
        """
        Constrained set defined by l2 norm ball.

        :param loc: Center of ball.
        :param scale: Radius of ball.
        """
        super().__init__()
        self.loc = loc.view(1, -1)
        self.r = scale
        self.r2 = scale * scale

    def check_feasibility(self, point: Tensor) -> bool:
        return torch.linalg.vector_norm(self.loc.to(point) - point) <= self.r

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        rel_p = (points - self.loc).to(device)
        return torch.einsum("ik, ik -> i", rel_p, rel_p) <= self.r2

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=None) -> Tensor:
        if device is None: device = os.device
        n = vs.shape[0]
        bs = os - self.loc.to(device)
        bb = torch.einsum("ik, ik -> i", bs, bs)

        fs = bb <= self.r2
        bb = bb[fs]
        bs = bs[fs]
        vs = vs[fs]

        vb = torch.einsum("ik, ik -> i", bs, vs)
        vv = torch.einsum("ik, ik -> i", vs, vs)
        delta = vb * vb - vv * (bb - self.r2)

        ts = torch.full([n], torch.inf, device=device)
        ts[fs] = (torch.sqrt(delta) - vb) / vv
        ts[ts < 0] = torch.inf
        return ts

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        return self.eval_intersection_v(o, v, tol, thresh)[0].item()


class QuadraticConstraint(ConstrainedSet):

    def __init__(self, Q: Tensor, p: Tensor, b: float):
        """
        Constrained set defined by l2 norm quadratic.

            x^T Q x + p^T x + b <= 0.

        :param Q: Quadratic term.
        :param p: 1D Tensor.
        :param b: Float constant.
        """
        super().__init__()
        self.Q = Q
        self.p = p.view(-1)
        self.b = b
        self.nb = -b

    def check_feasibility(self, point: Tensor) -> bool:
        return self.check_feasibility_v(point.view(1, -1))[0].item()

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        xQx = self._a_Q_b_(points, points)
        px = self._p_x_(points)
        return xQx + px <= self.nb

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        return self.eval_intersection_v(o.view(1, -1), v.view(1, -1), tol, thresh)[0].item()

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device()
    ):
        """
        Solve the equation (o+tv)^TQ(o+tv) + p^T(o+tv) + b == 0,
        which is equivalent to the following quadratic equation:

            v^TQv t^2 + (p^Tv+2o^TQv) t + (o^TQo + p^To + b) == 0

        :param os: The starting points.
        :param vs: Directions.
        :param tol: Values smaller than tolerance are considered 0.
        :param thresh: Ignored.
        :param device: Torch device.
        :return: 1D Tensor of float.
        """
        vQv = self._a_Q_b_(vs, vs).squeeze()
        oQo = self._a_Q_b_(os, os).squeeze()
        oQv = self._a_Q_b_(os, vs).squeeze()
        pv = self._p_x_(vs).squeeze()
        po = self._p_x_(os).squeeze()

        a = vQv
        b = pv + 2 * oQv
        c = oQo + po + self.b

        delta = b * b - 4 * a * c

        ii = torch.logical_or(delta < 0, torch.abs(a) < tol)
        delta[ii] = 0
        a[ii] = 1

        ts = (-b + torch.sqrt(delta)) / 2 / a
        ts[ii] = torch.inf

        return ts

    def _a_Q_b_(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.einsum("ik, kl, il -> i", a, self.Q.to(a), b)

    def _p_x_(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.p.to(x))


def ConeConstraint(A: Tensor, b: Tensor, c: Tensor, d: float) -> QuadraticConstraint:
    b = b.view(-1, 1)
    c = c.view(-1, 1)

    Q = A.T @ A - c @ c.T
    p = 2 * (b.T @ A - d * c.T)
    r = torch.sum(b * b).item() - d * d
    return QuadraticConstraint(Q, p, r)


class SemiDefiniteConstraint(ConstrainedSet):

    def __init__(self, Fs: Tensor):
        """
        Semi-definite constraint defined as
            F_0 + x_1 F_1 + \cdots + x_d F_d \succeq 0.
            X>0, [a,b]
        :param Fs: (d+1)*d*d Tensor, each Fs[i] should be symmetric.
        """
        super().__init__()
        self.F0 = Fs[0]
        self.Fs = Fs[1:]
        self.ndim = Fs.shape[0] - 1

    def check_feasibility(self, point: Tensor) -> bool:
        G = self.F0 + torch.sum(point.view(-1, 1, 1) * self.Fs, dim=0)
        eig = torch.linalg.eigvalsh(G)
        return torch.all(eig >= 0).item()

    def check_feasibility_v(self, points: Tensor, device=torch.get_default_device()) -> Tensor:
        Gs = self.F0.to(points) + torch.einsum("nk, kij -> nij", points, self.Fs.to(points))
        es = torch.linalg.eigvalsh(Gs)
        return torch.all(es >= 0, dim=1)

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        return self.eval_intersection_v(o.view(1, -1), v.view(1, -1), tol, thresh)[0].item()

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=None
    ):
        if device is None: device = os.device
        Hs = self.F0.to(os) + torch.einsum("nk, kij -> nij", os, self.Fs.to(os))
        Ss = torch.einsum("nk, kij -> nij", vs, self.Fs.to(vs))
        ls = torch.linalg.eigvalsh(-torch.matmul(torch.linalg.inv(Hs), Ss))[:, -1].to(device)
        ts = torch.full([vs.shape[0]], torch.inf, device=device)
        ts[ls > 0] = 1 / ls[ls > 0]
        return ts


class IneqConstraint(ConstrainedSet):

    def __init__(
            self,
            lhs: Callable[[Tensor], float],
            rhs: float = 0.0,
            direction: Literal["<=", "<", ">", ">="] = "<=",
    ):
        """
        Constrained set defined by an inequality.
        The equality is given by

            lhs `direction` rhs

        The left-hand side is assumed to be a function of Tensor that returns a float.

        The right-hand side is assumed to be a float constant, 0 by default.

        The direction is the relationship of lhs and rhs, "<=" by default.

        :param lhs: A function of Tensor that returns a float.
        :param rhs: A float constant, default 0.
        :param direction: Relationship of lhs and rhs, "<=" by default.
        """
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.dir = direction
        self.fun = _dir_func(direction)

    def check_feasibility(self, point: Tensor) -> bool:
        return self.fun(self.lhs(point), self.rhs)


class Intersection(ConstrainedSet):

    def __init__(self, *constraints: ConstrainedSet):
        """
        Intersection of some constrained sets, resulting to a more limiting constrained set.

        A point in this intersection set is feasible if and only if it is feasible for all given constrained sets.

        :param constraints: Constrained sets to create intersection from.
        """
        super().__init__()
        self.constraints = constraints

    def check_feasibility(self, point: Tensor) -> bool:
        return all([constraint.check_feasibility(point) for constraint in self.constraints])

    def eval_intersection(self, o: Tensor, v: Tensor, tol: float = 1e-6, thresh: float = 1e8) -> float:
        return min([c.eval_intersection(o, v, tol, thresh) for c in self.constraints])

    def eval_intersection_v(
            self,
            os: Tensor, vs: Tensor,
            tol: float = 1e-6, thresh: float = 1e8,
            device=torch.get_default_device()
    ):
        ts = torch.zeros([os.shape[0], len(self.constraints)], device=device)
        for i, c in enumerate(self.constraints):
            ts[:, i] = c.eval_intersection_v(os, vs, tol, thresh)
        return torch.min(ts, dim=1)[0]


def _dir_func(direction: Literal["<=", "<", ">", ">="]):
    match direction:
        case "<=":
            return _dir_le
        case "<":
            return _dir_ls
        case ">":
            return _dir_gt
        case ">=":
            return _dir_ge


def _dir_le(lhs, rhs):
    return lhs <= rhs


def _dir_ls(lhs, rhs):
    return lhs < rhs


def _dir_gt(lhs, rhs):
    return lhs > rhs


def _dir_ge(lhs, rhs):
    return lhs >= rhs
