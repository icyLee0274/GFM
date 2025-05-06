import torch
from torch import Tensor
from torch.distributions import Distribution as TorchDistribution, constraints as torch_constraints
import itertools
from .constraints import ConstrainedSet

__all__ = [
    "HyperBallUniform",
    "HyperBoxUniform",
    "box_uniform",
    "SumDistribution",
    "TruncatedDistribution",
]


class HyperBallUniform(TorchDistribution):
    arg_constraints = {}

    def __init__(self, dim: int | None = None, loc: Tensor | None = None, scale: float | None = None):
        """
        Creates a uniform distribution in a hyper-ball of dimension `dim`.
        The center and radius of the ball are parameterized by `loc` and `scale`.

        :param dim: Dimension of the distribution, leave it to `None` so to inference from `loc`.
        :param loc: Location parameter of the distribution, if `None`, set to 0
        :param scale: Scale parameter of the distribution, if `None`, set to 1
        """
        loc = torch.flatten(loc) if loc is not None else None
        if dim is not None:
            self.dim = dim
            if loc is not None:
                assert dim == len(loc)
                self.loc = loc
            else:
                self.loc = torch.zeros(dim)
        elif loc is not None:
            self.dim = len(loc)
            self.loc = loc
        else:
            raise ValueError("Must specify either loc or dim")
        if scale <= 0: raise ValueError("Scale must be positive")
        self.scale = scale if scale is not None else 1
        super().__init__(event_shape=loc.shape)

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        directions = torch.randn(shape)
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        u = torch.rand([*sample_shape, 1])
        r = u ** (1.0 / self.dim)
        return directions * r * self.scale + self.loc


class HyperBoxUniform(TorchDistribution):
    arg_constraints = {"loc": torch_constraints.real, "scale": torch_constraints.positive}

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        super().__init__(event_shape=loc.shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.rand(shape) * self.scale + self.loc


def box_uniform(loc: Tensor | int = 1, scale: float | Tensor = 1.) -> HyperBoxUniform:
    """
    A wrapper around `HyperBoxUniform` that creates a uniform distribution in a hyper-box centered at `loc` with radius `scale`.
    The set is
    \[
        \{ x | loc - scale < x < loc + scale \}.
    \]

    :param loc: Location parameter of the distribution, if an `int`, zeros of dimension `loc`.
    :param scale: Scale parameter of the distribution, if an `float`, set to a vector of dimension same as `loc` with all elements equal to `scale`.
    :return: A `HyperBoxUniform` distribution with properly justified `loc` and `scale`.
    """
    if loc is int:
        loc = torch.zeros(loc)
    if scale is float:
        scale = torch.full_like(loc, scale)
    loc = torch.flatten(loc)
    scale = torch.flatten(scale)
    assert loc.shape == scale.shape
    loc = loc - scale
    scale = 2 * scale
    return HyperBoxUniform(loc, scale)


class SumDistribution(TorchDistribution):
    arg_constraints = {}
    _one = torch.Size((1,))

    def __init__(
            self,
            *distributions: TorchDistribution,
            weights: Tensor = None,
            device: torch.device = torch.get_default_device()
    ):
        """
        Sum of some distributions.

            q = \sum_{i=1}^{n} w_i p_i

        :param distributions: Distributions to be summed.
        :param weights: Weight of each distribution, defaults to 1.
        :param device: Torch device.
        """
        es = distributions[0].event_shape
        super().__init__(event_shape=es)
        if weights is None:
            weights = torch.ones(len(distributions))
        self.distributions = distributions
        self.weights = weights
        self.ig = torch.distributions.Categorical(weights)
        self.device = device

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        choices = self.ig.sample(sample_shape)
        samples = torch.zeros(shape, device=self.device)
        for index in itertools.product(*[range(i) for i in sample_shape]):
            samples[index] = self.distributions[choices[index]].rsample(self._one)
        return samples


class TruncatedDistribution(TorchDistribution):
    arg_constraints = {}
    _one = torch.Size((1,))

    def __init__(
            self,
            distribution: TorchDistribution,
            constraint: ConstrainedSet,
            device: torch.device = torch.get_default_device()
    ):
        """
        Truncated distribution constrained by the provided constrained set.
        All thereafter generated samples are feasible with respect to the constrained set.

        :param distribution: Distribution to be truncated.
        :param constraint: Constrained set.
        :param device: Torch device.
        """
        super().__init__(event_shape=distribution.event_shape)
        self.distribution = distribution
        self.constraint = constraint
        self.device = device

    def _sample_one(self):
        while True:
            sample = self.distribution.rsample(self._one)
            if self.constraint.check_feasibility(sample):
                return sample

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        samples = torch.zeros(shape, device=self.device)
        for index in itertools.product(*[range(i) for i in sample_shape]):
            samples[index] = self._sample_one()
        return samples
