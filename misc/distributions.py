import torch
from torch import Tensor
from torch.distributions import Distribution as TorchDistribution, constraints as torch_constraints
import itertools
from .constraints import ConstrainedSet

__all__ = ["SumDistribution", "TruncatedDistribution", "HyperBoxUniform", "UnitBallUniform"]


class UnitBallUniform(TorchDistribution):
    arg_constraints = {}

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        directions = torch.randn([*sample_shape, self.dim])
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        u = torch.rand([*sample_shape, 1])
        r = u ** (1.0 / self.dim)
        return directions * r


class HyperBoxUniform(TorchDistribution):
    arg_constraints = {"loc": torch_constraints.real, "scale": torch_constraints.positive}

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        super().__init__(event_shape=loc.shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        return torch.rand(shape) * self.scale + self.loc


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
