import emcee
import torch

from torch.autograd.functional import hessian


def initialize_backend(fn_backend: str):
    """Initialize the backend for the MCMC sampler."""
    return emcee.backends.HDFBackend(fn_backend)


def initialize_walkers(
    priors: dict, n_walkers: int, specs: object = None
) -> torch.Tensor:

    """
    Initialize walkers for the MCMC sampler.

    Parameters
    ----------
    priors : dict
        Dict of prior distributions for each parameter.
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
        means = torch.tensor([p.mean for p in priors.values()])
        stds = torch.tensor([p.scale for p in priors.values()])
        p0 = torch.normal(means.expand(n_walkers, -1), stds.expand(n_walkers, -1))
    else:
        n_params = len(specs.bounds_implicit.bounds)
        n_dim = len(priors)

        p0 = torch.empty((n_walkers, n_dim))
        count = 0
        while count < n_walkers:
            p0_trial = torch.tensor([p.sample().item() for p in priors.values()])
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

    params = check_tensor(params, device=lbe.device)

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


# Global toggle for manual squared distance
_MANUAL_MODE = False


class enable_manual_dist:
    """Context manager to enable manual pairwise distance computation (Hessian-safe)."""
    def __enter__(self):
        global _MANUAL_MODE
        self._prev = _MANUAL_MODE
        _MANUAL_MODE = True

    def __exit__(self, *args):
        global _MANUAL_MODE
        _MANUAL_MODE = self._prev


def auto_manual_switch(fn):
    def wrapper(*args, manual_sqdist=False, **kwargs):
        return fn(*args, manual_sqdist=manual_sqdist or _MANUAL_MODE, **kwargs)
    return wrapper


def find_max_stable_lr(
    fn: callable,
    p0: torch.Tensor,
    learning_rates: float | torch.Tensor = None,
    max_iter: int = 200,
    param_bounds: tuple[float, float] = (-7, 7),
) -> float | None:
    """Find the largest stable learning rate for gradient-based optimization.

    Parameters
    ----------
    fn : Callable
        Objective function returning a scalar tensor.
    p0 : torch.Tensor
        Initial parameter vector.
    learning_rates : iterable of float, optional
        Learning rates to test. Defaults to log-spaced values.
    max_iter : int
        Number of steps to test for each learning rate.
    param_bounds : tuple of float
        Bounds beyond which parameters are considered unstable.

    Returns
    -------
    float or None
        The largest stable learning rate found, or None if none were stable.
    """
    if learning_rates is None:
        learning_rates = 10 ** torch.linspace(-1, -6, 6)
    lower, upper = param_bounds
    for lr in learning_rates:
        x = p0.clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([x], lr=lr)

        for i in range(max_iter):
            if torch.any(x < lower) or torch.any(x > upper):
                break
            opt.zero_grad()
            loss = -fn(x)
            loss.backward()
            opt.step()
        else:
            # Only gets executed if inner loop did not break (i.e., stable)
            return lr


def find_map(
    fn: callable,
    x0: torch.Tensor,
    lr: float | torch.Tensor = None,
    max_iter: int = 10000,
    tol_grad: float = 1e-2,
    device: str = 'cpu',
    logger: callable = None
):

    logger.info('  > optimizing hyperparameters: stable learning rate search: in progres...', overwrite=True)
    lr_opt = 0.5 * find_max_stable_lr(fn, x0, learning_rates=lr)
    if lr_opt is not None:
        logger.info(f'  > optimizing hyperparameters: stable learning rate search: Done. | {lr_opt:.1e}')
    else:
        raise ValueError('No stable learning rate found.')

    x0 = x0.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.SGD([x0], lr=lr_opt)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -fn(x0)
        loss.backward()
        grad_norm = x0.grad.norm().item()
        if i % 100 == 0 and logger is not None:
            logger.info(f"  > optimizing hyperparameters: it. {i}/{max_iter} | loss: {loss.item():.3f} | grad: {grad_norm:.3f}/{tol_grad}", overwrite=True)
        optimizer.step()
        if grad_norm < tol_grad:
            logger.info('  > optimizing hyperparameters: Done.')
            break
    
    else:
        logger.info('  > optimizing hyperparameters: Fail. | Max iterations reached without convergence.')

    return x0.detach()


def laplace_approximation(
    fn: callable,
    map_theta: torch.Tensor,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Laplace approximation around MAP estimate.

    Returns
    -------
    mean : torch.Tensor
        The MAP estimate (same as input).
    cov : torch.Tensor
        The approximate posterior covariance (inverse Hessian).
    """

    map_theta = check_tensor(map_theta, device=device)
    with enable_manual_dist():
        H = -hessian(lambda th: fn(th).sum(), map_theta)

    reg_eye = 1e-6 * torch.eye(H.shape[0], device=H.device)
    cov = torch.linalg.inv(H + reg_eye)

    # symmerize covariance matrix
    cov = (cov + cov.T) / 2

    return cov
