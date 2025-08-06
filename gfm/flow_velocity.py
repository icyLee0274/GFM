import torch
from torch import nn, Tensor

__all__ = ["FlowVelocity", "Mlp", "ResNet"]


class VelocityFieldBase(nn.Module):

    def __init__(self, net: nn.Module):
        """
        Base class for velocity fields.
        """
        super().__init__()
        self.net = net

    def forward(self, *args: Tensor) -> Tensor:
        """
        Forward method to compute the velocity field.

        The input arguments are assumed to be time and space points, in form of tensors.
        The input tensor can be either
            1. Tensor of shape [n,d], where n is the batched size and d is the dimension of the space.
            2. Tensor of shape [n,1], where n is the batched size.
            3. Tensor of shape [n], where n is the batched size.
            4. Tensor of shape [1], which is a single time point.
            5. Tensor of shape [], which is a scalar time point.
        These tensors are expanded to proper shape, i.e., (n,1) for case 3, 4 and 5, if applicable,
        concatenated, and passed to the underlying neural network, i.e., ``VelocityFieldBase.net``.

        ----
        :param args: Input tensors, typically time and state.
        :return: Computed velocity field.
        """
        n = max([arg.shape[0] for arg in args if arg.dim() > 0], default=1)
        for arg in args:
            if arg.dim() > 0 and arg.shape[0] != n and arg.shape[0] != 1:
                raise ValueError("All batched input tensors must have the same first dimension.")
        args = [arg if arg.dim() > 1 else
                arg.expand(n, 1) if arg.dim() == 0 or (arg.dim() == 1 and arg.shape[0] == 1) else
                arg.view(n, 1) for arg in args]
        x = torch.cat(args, dim=1)  # Concatenate all inputs along the last dimension
        return self.net(x)


class Mlp(VelocityFieldBase):

    def __init__(
            self,
            n_in: int = 1,
            n_out: int = 2,
            width: int = 64,
            depth: int = 2,
            activation: str = "ELU"
    ):
        """
        Multi-layer perceptron (MLP) for modeling the velocity field.

        :param n_in: Number of input dimensions.
        :param n_out: Number of output dimensions.
        :param width: Width of hidden layers.
        :param depth: Number of hidden layers.
        :param activation: Activation function.
        """
        activation = getattr(nn, activation)
        hidden = [nn.Linear(n_in, width), activation()]
        for _ in range(depth):
            hidden.append(nn.Linear(width, width))
            hidden.append(activation())
        hidden.append(nn.Linear(width, n_out))
        net = nn.Sequential(*hidden)
        super().__init__(net)


class _ResBlock(nn.Module):

    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid),
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in), )

    def forward(self, x):
        return x + self.net(x)


class ResNet(VelocityFieldBase):

    def __init__(self, n_in, n_out, n_hidden, n_layers):
        """
        Residual neural network (ResNet) for modeling the velocity field.

        :param n_in: Number of input dimensions.
        :param n_out: Number of output dimensions.
        :param n_hidden: Width of hidden layers.
        :param n_layers: Number of res-net blacks.
        """
        blocks = [nn.Linear(n_in, n_hidden)]
        for _ in range(n_layers):
            blocks += [_ResBlock(n_hidden, n_hidden // 2)]  # nh // 2, nh // 4, nh // 8
        blocks.append(nn.Linear(n_hidden, n_out))

        net = nn.Sequential(*blocks)

        super().__init__(net)


class FlowVelocity(nn.Module):
    """
    This class implements a two-layer MLP for modeling the flow velocity field.

    It is kept for backward compatibility with the original implementation.
    Please use ``Mlp`` or ``ResNet`` instead.

    This class is subject to deprecation and will be removed in future versions.
    """

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        return self.net(torch.cat((t, x_t), dim=-1))


class FlowVelocityDeep3(nn.Module):
    """
    This class implements a three-layer MLP for modeling the flow velocity field.

    It is kept for backward compatibility with the original implementation.
    Please use ``Mlp`` or ``ResNet`` instead.

    This class is subject to deprecation and will be removed in future versions.
    """

    def __init__(self, dim: int = 2, h: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(len(x_t), 1)
        return self.net(torch.cat((t, x_t), dim=-1))
