import torch
from .utils import check_tensor, auto_manual_switch


@auto_manual_switch
def gaussian_kernel(
    x1: torch.Tensor, x2: torch.Tensor,
    lengths: float | torch.Tensor, width: float | torch.Tensor,
    manual_sqdist: bool = False
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
    manual_sqdist : bool, optional
        If True, computes the squared distance manually.
        If False, uses `torch.cdist` for efficiency.
        manual_sqdist is necessary for evaluation of second derivatives.

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
        x1 = x1[None, :, :].to(device)
    if x2.dim() == 2:
        x2 = x2[None, :, :].to(device)

    lengths = check_tensor(lengths, device=device)
    if lengths.dim() == 1:
        lengths = lengths[None, :].to(device)

    width = check_tensor(width, device=device)
    if width.dim() == 1:
        width = width[None, :].to(device)

    x1_scaled = x1 / lengths
    x2_scaled = x2 / lengths

    if manual_sqdist:
        x1_sq = (x1_scaled**2).sum(dim=-1, keepdim=True)  # (B, N1, 1)
        x2_sq = (x2_scaled**2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N2)
        sqdist = x1_sq + x2_sq - 2 * (x1_scaled @ x2_scaled.transpose(1, 2))
        sqdist = torch.clamp(sqdist, min=0.0)
    else:
        sqdist = torch.cdist(x1_scaled, x2_scaled, p=2).pow(2)

    kernel = width.pow(2).mean() * torch.exp(-0.5 * sqdist)

    return kernel.squeeze(0)
