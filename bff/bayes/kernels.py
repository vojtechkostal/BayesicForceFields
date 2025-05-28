import torch


def gaussian_kernel(
    x1: torch.Tensor, x2: torch.Tensor,
    lengths: float | torch.Tensor, width: float | torch.Tensor
) -> torch.Tensor:
    """Gaussian (Squared Exponential) kernel.

    Parameters
    ----------
    x1 : torch.Tensor
        First input tensor of shape (n_samples, n_features) or (n_features,).
    x2 : torch.Tensor
        Second input tensor of shape (n_samples, n_features) or (n_features,).
    lengths : float or torch.Tensor
        Length scale(s) for the kernel.
        Can be a single float or a tensor of shape (n_features,).
    width : float or torch.Tensor
        Width scale(s) for the kernel.
        Can be a single float or a tensor of shape (n_features,).

    Returns
    -------
    torch.Tensor
        Kernel matrix of shape (n_samples_x1, n_samples_x2).
        If x1 and x2 are 2D tensors, the output will be of shape
        (n_samples, n_samples_x1, n_samples_x2).
    """

    assert x1.device == x2.device, "x1 and x2 must be on the same device"
    device = x1.device

    if x1.dim() == 2:
        x1 = x1[torch.newaxis, :, :].to(device)
    if x2.dim() == 2:
        x2 = x2[torch.newaxis, :, :].to(device)

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor([lengths])
    if lengths.dim() == 1:
        lengths = lengths[torch.newaxis, :].to(device)
    if not isinstance(width, torch.Tensor):
        width = torch.tensor([width])
    if width.dim() == 1:
        width = torch.tensor([width])[torch.newaxis, :].to(device)

    pairwise_disances = torch.cdist(x1 / lengths, x2 / lengths, p=2)
    kernel = width**2 * torch.exp(- 0.5 * pairwise_disances**2)

    return kernel.squeeze(0)
