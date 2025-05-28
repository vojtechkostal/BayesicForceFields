import emcee
import torch


def initialize_backend(fn_backend: str):
    """Initialize the backend for the MCMC sampler."""
    return emcee.backends.HDFBackend(fn_backend)


def initialize_walkers(
    priors: list, n_walkers: int, specs: object = None
) -> torch.Tensor:

    """
    Initialize walkers for the MCMC sampler.

    Parameters
    ----------
    priors : list
        List of prior distributions for each parameter.
    n_walkers : int
        Number of walkers to initialize.
    specs : object, optional
        Specifications object containing bounds and constraints.
        If provided, walkers will be initialized within the bounds.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_walkers, n_params) containing the initial positions
        of the walkers, sampled from the prior distributions.
    """

    if not specs:
        means = torch.tensor([p.mean for p in priors])
        stds = torch.tensor([p.scale for p in priors])
        p0 = torch.normal(means.expand(n_walkers, -1), stds.expand(n_walkers, -1))
    else:
        n_params = len(specs.bounds_implicit.bounds)
        n_dim = len(priors)

        p0 = torch.empty((n_walkers, n_dim))
        count = 0
        while count < n_walkers:
            p0_trial = torch.tensor([p.sample().item() for p in priors])
            if valid_bounds(p0_trial[:n_params].unsqueeze(0), specs):
                p0[count] = p0_trial
                count += 1
    return p0


def valid_bounds(params: torch.Tensor, specs: object) -> torch.Tensor:
    """
    Check if all the samples fall within explicit and implicit constraints.
    Assumes params is a 2D torch tensor of shape (batch_size, n_params)
    """
    lbe, ube = torch.tensor(specs.bounds_implicit.values).T
    lbi, ubi = torch.tensor(specs.implicit_param_bounds)

    params = params.to(dtype=torch.float32, device=lbe.device)

    valid_explicit = ((params > lbe) & (params < ube)).all(dim=1)

    constraint_matrix = torch.tensor(
        specs.constraint_matrix, dtype=params.dtype, device=params.device)
    q_explicit = torch.sum(params * constraint_matrix, dim=1)
    q_implicit = specs.total_charge - q_explicit
    valid_implicit = (q_implicit >= lbi) & (q_implicit <= ubi)

    return valid_explicit & valid_implicit


def check_tensor(x, device):
    """Convert input to a torch tensor on the specified device."""
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device, dtype=torch.float32)
    else:
        return x.to(device, dtype=torch.float32)


def check_device(device):
    """Check if the specified device is available."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
    elif device.startswith("mps"):
        if not torch.mps.is_available():
            raise RuntimeError("MPS is not available.")


@torch.no_grad()
def nearest_positve_definite(A):
    """Find the nearest positive definite matrix to A."""
    # Symmetrize
    A_sym = (A + A.T) / 2

    # Check if already PD
    if torch.all(torch.linalg.eigvalsh(A_sym) > 0):
        return A_sym

    # Eigen-decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

    # Shift eigenvalues minimally
    min_eig = eigenvalues.min()
    eps = 1e-8  # small positive shift

    if min_eig < eps:
        eigenvalues = eigenvalues - min_eig + eps

    A_pd = (eigenvectors @ torch.diag(eigenvalues)) @ eigenvectors.T
    return A_pd


def train_test_split(
    X: torch.Tensor, y: torch.Tensor, test_fraction: float = 0.2
) -> tuple:
    """Split the dataset into training and testing sets."""

    n = len(X)
    if n != len(y):
        raise ValueError("X and y must have the same length.")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1.")
    if n < 2:
        raise ValueError("X and y must have at least 2 samples.")

    indices = torch.randperm(n)
    test_size = int(n * test_fraction)
    idx_train = indices[test_size:]
    idx_test = indices[:test_size]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]
