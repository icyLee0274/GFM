import torch
from torch import Tensor, tensor
from .constraints import ConstrainedSet


class MetropolisSampler:

    def __init__(self, domain: ConstrainedSet):
        self.domain = domain

    def __call__(self, os: Tensor, vs: Tensor) -> Tensor:
        ds = os + vs
        fs = self.domain.check_feasibility_v(ds)
        while not fs.all():
            # Sample a new point
            _vs = torch.randn_like(vs[~fs])
            _vs = (_vs / torch.linalg.vector_norm(_vs, dim=-1, keepdim=True) *
                   torch.linalg.vector_norm(vs[~fs], dim=-1, keepdim=True))
            vs[~fs] = _vs
            ds = os + vs
            fs = self.domain.check_feasibility_v(ds)
        return ds
