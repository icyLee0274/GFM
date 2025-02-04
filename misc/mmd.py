import torch


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
    n = xs.shape[1]
    uxx = -m
    uyy = -n
    uxy = 0
    for i in range(m):
        uxx += torch.sum(gaussian_kernel(xs[i], xs))
        uxy += torch.sum(gaussian_kernel(xs[i], ys))
    for j in range(n):
        uyy += torch.sum(gaussian_kernel(ys[j], ys))
    return uxx / m / (m - 1) + uyy / n / (n - 1) - 2 * uxy / m / n
