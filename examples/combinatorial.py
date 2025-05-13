import torch
from torch import Tensor

from itertools import product
from math import exp

import gfm

__all__ = ["CombinatorialSampler", "s_vec", "s_vec_inv"]


def s_vec(x: Tensor) -> Tensor:
    """
    Vectorize a symmetric matrix.

    :param x: d*d matrix
    :return: d*(d+1)/2 vector
    """
    dim = x.shape[0]
    indices = torch.triu_indices(dim, dim, 0)
    return x[indices[0], indices[1]]


def s_vec_inv(x: Tensor) -> Tensor:
    """
    Inverse vectorization of a symmetric matrix.

    :param x: d*(d+1)/2 vector
    :return: d*d matrix
    """
    dim = int((-1 + (1 + 8 * x.shape[0]) ** 0.5) / 2)
    indices = torch.triu_indices(dim, dim, 0)
    y = torch.zeros(dim, dim, dtype=x.dtype, device=x.device)
    y[indices[0], indices[1]] = x
    return y + y.T - torch.diag(y.diagonal())


class CombinatorialSampler:

    def __init__(
            self,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            Ds: Tensor,
            cs: Tensor,
    ):
        """
        Matrix sampler that generates d*d matrices from a distribution with the following density:

        .. math::
            \exp\{ -(<A,X> + \|X-B\|_F + \|X-C\|^2_F - \log\det X) \}

            s.t. X \succ 0

            <D_i, X> \ge c_i, i=1,...,m

        where
            - :math:`\|X\|_F` is the Frobenius norm
            - :math:`<A,X>` is the Frobenius inner product of matrices
            - :math:`\log\det X` is the log-determinant of the matrix

        :param A: d*d matrix
        :param B: d*d matrix
        :param C: d*d matrix
        :param Ds: m*d*d matrix
        :param cs: m(*1) vector
        """
        self.A = A
        self.B = B
        self.C = C
        self.Ds = Ds
        self.cs = cs
        self.dim = A.shape[0]

        k = 0
        Fs = torch.zeros(self.dim + 1, self.dim, self.dim, device=A.device)
        for i in range(0, self.dim):
            for j in range(i, self.dim):
                Fs[k, i, j] = 1
                Fs[k, j, i] = 1
                k += 1
        self.psd = gfm.SemiDefiniteConstraint(Fs)

        indices = torch.triu_indices(self.dim, self.dim, 0)
        _Ds = torch.zeros(Ds.shape[0], int(self.dim * (self.dim + 1) / 2), device=A.device)
        for i in range(Ds.shape[0]): _Ds[i] = s_vec(-2 * Ds[i] + torch.diag(_Ds[i].diagonal()))
        self.linear = gfm.LinearConstraint(_Ds, -cs)

        self.acceptance_rate = 0

    def sample(
            self,
            X_0: Tensor,
            n_samples: int,
            burn_in: int = 1000,
            thin: int = 1,
            atol: float = 1e-5,
            boundary_tol: float = 1e-6,
            metro_scale: float = 0.1,
            metro_burn: int = 100,
            seed=None,
    ) -> Tensor:
        """
        Sample from the distribution.

        :param X_0: Initial symmetric, positive definite matrix that satisfies the constraint
        :param n_samples: Number of samples to generate
        :param burn_in: Number of samples to discard
        :param thin: Thinning factor
        :param atol: Tolerance
        :param boundary_tol: Boundary tolerance
        :param metro_scale: Scale for the Metropolis-Hastings proposal
        :param metro_burn: Number of burn-in samples for the Metropolis-Hastings proposal
        :param seed: Random seed
        :return: n_samples*d*d matrix
        """

        device = X_0.device
        dim = self.dim
        X_curr = X_0.clone()
        total_iter = burn_in + n_samples * thin
        accepted_moves_post_burn_in = 0
        n_generated = 0
        null = torch.zeros_like(X_0)
        samples = torch.empty(n_samples, dim, dim, device=device)

        rng = torch.Generator(device)
        if seed is not None: rng.manual_seed(seed)

        for i in range(total_iter):

            # 1. Generate a random symmetric unit matrix
            D = torch.randn(dim, dim, generator=rng, device=device)
            D = (D + D.T) / 2
            D_norm = torch.linalg.matrix_norm(D, ord='fro')
            if D_norm < atol:
                D = torch.zeros_like(D)
            else:
                D = D / D_norm
            if torch.allclose(D, null, atol=atol):
                if i >= burn_in and (i - burn_in) % thin == 0:
                    samples[n_generated] = X_curr
                    n_generated += 1
                if n_generated >= n_samples: break
                continue

            # 2. Find lambda bounds
            lambda_min, lambda_max = self.lambda_bounds(X_curr, D)
            if lambda_min is None or lambda_max < lambda_min + boundary_tol:
                if i >= burn_in and (i - burn_in) % thin == 0:
                    samples[n_generated] = X_curr
                    n_generated += 1
                if n_generated >= n_samples: break
                continue

            # 3. Generate a lambda
            lambda_next = self.sample_lambda(X_curr, D, lambda_min, lambda_max, metro_scale, metro_burn, rng)
            X_next = X_curr + lambda_next * D

            # 4. Check acceptance; I don't think this will fail
            if self.check_feasibility(X_next):
                X_curr = X_next
                accepted_moves_post_burn_in += 1

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples[n_generated] = X_curr
                n_generated += 1
            if n_generated >= n_samples: break

        self.acceptance_rate = accepted_moves_post_burn_in / (total_iter - burn_in)

        return samples

    def lambda_bounds(self, X_curr, D):
        X_vec = s_vec(X_curr)
        D_vec = s_vec(D)

        t_spd_max = self.psd.eval_intersection(X_vec, D_vec)
        t_spd_min = self.psd.eval_intersection(X_vec, -D_vec)
        t_linear_max = self.linear.eval_intersection(X_vec, D_vec)
        t_linear_min = self.linear.eval_intersection(X_vec, -D_vec)

        return max(t_spd_min, t_linear_min), max(t_spd_max, t_linear_max)

    def check_feasibility(self, X_next) -> bool:
        X_vec = s_vec(X_next)
        return self.psd.check_feasibility(X_vec) and self.linear.check_feasibility(X_vec)

    def density(self, X: Tensor) -> float:
        a = torch.sum(X * self.A.to(X)).item()
        b = torch.linalg.matrix_norm(X - self.B.to(X)).item()
        c = torch.linalg.matrix_norm(X - self.C.to(X)).item() ** 2
        d = torch.log(torch.det(X)).item()
        return exp(-(a + b + c - d))

    def sample_lambda(
            self,
            X_curr,
            D,
            lambda_min, lambda_max,
            scale,
            burn_in,
            generator):
        l_curr = 0.
        p_curr = self.density(X_curr)
        for _ in range(burn_in + 1):
            l_next = l_curr + scale * torch.randn(1, generator=generator).item()
            if l_next < lambda_min or l_next > lambda_max: continue
            p_next = self.density(X_curr + l_next * D)
            ratio = p_next / p_curr
            if torch.rand(1, generator=generator).item() < ratio:
                l_curr = l_next
                p_curr = p_next
        return X_curr + l_curr * D
