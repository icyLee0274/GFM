import torch
from torch import tensor, Tensor
from torchdiffeq import odeint


def odeint_reflect(
        func,
        y0,
        t,
        rtol=1e-7,
        atol=1e-9,
        method=None,
        options=None,
        event_fn=None,
        reflect_fn=None,
):
    """Integrate a system of reflecting ordinary differential equations.

        Solves the initial value problem for a non-stiff system of first order reflecting ODEs:
            ```
            dy/dt = func(t, y) + L(t), y(t[0]) = y0
            ```
        where y is a Tensor or tuple of Tensors of any shape.

        Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

        Args:
            func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
                into a Tensor of state derivatives with respect to time. Optionally, `y`
                can also be a tuple of Tensors.
            y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
                can also be a tuple of Tensors.
            t: 1-D Tensor holding a sequence of time points for which to solve for
                `y`, in either increasing or decreasing order. The first element of
                this sequence is taken to be the initial time point.
            rtol: optional float64 Tensor specifying an upper bound on relative error,
                per element of `y`.
            atol: optional float64 Tensor specifying an upper bound on absolute error,
                per element of `y`.
            method: optional string indicating the integration method to use.
            options: optional dict of configuring options for the indicated integration
                method. Can only be provided if a `method` is explicitly set.
            event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
                event_fn evaluates to zero. If this is not None, all but the first elements of
                `t` are ignored.
            reflect_fn: Function that maps a tensor of starting points and a tensor of directions
                to reflecting end points.

        Returns:
            y: Tensor, where the first dimension corresponds to different
                time points. Contains the solved value of y for each desired time point in
                `t`, with the initial value `y0` being the first element along the first
                dimension.

        Raises:
            ValueError: if an invalid `method` is provided.
        """
    n_step = len(t) - 1
    x_t = torch.empty([n_step + 1, y0.shape[0], y0.shape[1]], device=y0.device)
    x_t[0] = y0
    for step in range(n_step):
        x_step = odeint(
            func, x_t[step], t[[step, step + 1]],
            rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn
        )[-1]
        x_t[step + 1] = reflect_fn(x_t[step], x_step - x_t[step])
    return x_t
