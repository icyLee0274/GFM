import torch
from torch import tensor, Tensor
from torchdiffeq import odeint

__all__ = ['odeint_reflect']


def odeint_reflect(
        func,
        y0,
        t,
        reflect_fn=None,
):
    """
        Integrate a system of reflecting ordinary differential equations by mid-point method.

        Solves the initial value problem for a non-stiff system of first order reflecting ODEs:
            ```
            dy/dt = func(t, y) + L(t), y(t[0]) = y0
            ```
        where y is a Tensor or tuple of Tensors of any shape.

        The iteration goes like
        \[
            x_1 = x_0 + (t_1 - t_0) * func( (t_1 + t_0) / 2, x_0 + (t_1 + t_0) / 2 * func(t_0, x_0) ).
        \]

        Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

        Args:
            func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
                into a Tensor of state derivatives with respect to time. Optionally, `y`
                can also be a tuple of Tensors.
            y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
                can also be a tuple of Tensors.
            T: 1-D Tensor holding a sequence of time points for which to solve for
                `y`, in either increasing or decreasing order. The first element of
                this sequence is taken to be the initial time point.
            reflect_fn: Function that maps a tensor of starting points and a tensor of directions
                to reflecting end points.

        Returns:
            y: Tensor, where the first dimension corresponds to different
                time points. Contains the solved value of y for each desired time point in
                `t`, with the initial value `y0` being the first element along the first
                dimension.
        """
    n_step = len(t) - 1
    x_t = torch.empty([n_step + 1, y0.shape[0], y0.shape[1]], device=y0.device)
    x_t[0] = y0
    for step in range(n_step):
        t0 = t[step]
        t1 = t[step + 1]
        x_step = x_t[step] + (t1 - t0) * func(t0, x_t[step])
        if reflect_fn is not None:
            x_t[step + 1] = reflect_fn(x_t[step], x_step - x_t[step])
        else:
            x_t[step + 1] = x_step
    return x_t
