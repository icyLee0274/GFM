import torch

__all__ = ['mmd_square', 'gaussian_kernel', 'maximum_mean_discrepancy', 'compute_mmd']


def gaussian_kernel(
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float = 1.0,
) -> torch.Tensor:
    """
    Computes the Gaussian kernel (RBF) between two sets of points.

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples_x, n_features).
        y (torch.Tensor): Input tensor of shape (n_samples_y, n_features).
        sigma (float): Bandwidth parameter (standard deviation) of the Gaussian kernel.

    Returns:
        torch.Tensor: Kernel matrix of shape (n_samples_x, n_samples_y).
    """
    # Ensure inputs are at least 2D for batch processing
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Compute squared Euclidean distances between all pairs
    squared_dist = torch.cdist(x, y, p=2).pow(2)

    # Compute gamma (1 / (2 * sigma^2))
    gamma = 1.0 / (2.0 * sigma ** 2)

    # Compute Gaussian kernel
    kernel = torch.exp(-gamma * squared_dist)

    # Squeeze to scalar if inputs were single vectors
    if kernel.size(0) == 1 and kernel.size(1) == 1:
        kernel = kernel.squeeze()

    return kernel


def mmd_square(xs: torch.Tensor, ys: torch.Tensor, sigma: float = 1.0) -> float:
    m = xs.shape[0]
    n = ys.shape[0]
    uxx = gaussian_kernel(xs, xs, sigma).sum() - m
    uxy = gaussian_kernel(xs, ys, sigma).sum()
    uyy = gaussian_kernel(ys, ys, sigma).sum() - n
    return uxx / m / (m - 1) + uyy / n / (n - 1) - 2 * uxy / m / n


def maximum_mean_discrepancy(xs: torch.Tensor, ys: torch.Tensor, sigma: float = 1.0) -> float:
    return mmd_square(xs, ys, sigma) ** 0.5


def compute_mmd(xs: torch.Tensor, ys: torch.Tensor, sigma: float = 1.0) -> float:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        xs (torch.Tensor): Samples from distribution X, shape (n_samples_x, n_features).
        ys (torch.Tensor): Samples from distribution Y, shape (n_samples_y, n_features).
        sigma (float): Bandwidth for the Gaussian kernel.

    Returns:
        float: The MMD value.
    """
    m = xs.shape[0]
    n = ys.shape[0]
    k_xx = gaussian_kernel(xs, xs, sigma)
    k_yy = gaussian_kernel(ys, ys, sigma)
    k_xy = gaussian_kernel(xs, ys, sigma)

    mmd_sq = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    return mmd_sq.sqrt().item()
