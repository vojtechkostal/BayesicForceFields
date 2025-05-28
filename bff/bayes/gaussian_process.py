import torch
from .kernels import gaussian_kernel
from .utils import check_tensor, nearest_positve_definite

from ..structures import Specs


class LocalGaussianProcess:
    """Local Gaussian Process Regression."""

    def __init__(
        self,
        X_train: torch.Tensor, y_train: torch.Tensor, y_mean: torch.Tensor,
        lengths: torch.Tensor, width: float, sigma: float,
        device: str
    ) -> None:

        self.X_train = check_tensor(X_train, device=device)
        self.y_train = check_tensor(y_train, device=device)
        self.y_mean = check_tensor(y_mean, device=device)
        self.lengths = check_tensor(lengths, device=device)
        self.width = check_tensor(width, device=device)
        self.sigma = check_tensor(sigma, device=device)
        self.device = device

        n_samples = len(self.X_train)

        noise = torch.eye(n_samples, device=device) * self.sigma
        Kdd = gaussian_kernel(self.X_train, self.X_train, self.lengths, width) + noise
        Kdd = nearest_positve_definite(Kdd)
        L = torch.linalg.cholesky(Kdd)
        Kdd_inv = torch.cholesky_inverse(L)
        self.Kdd_inv = check_tensor(Kdd_inv, device=device)

    @property
    def n_params(self):
        return self.X_train.shape[1]

    @property
    def y_size(self):
        return self.y_train.shape[1]

    @property
    def hyperparameters(self):
        return {
            'lengths': self.lengths.cpu().numpy(),
            'width': self.width.cpu().numpy().item(),
            'sigma': self.sigma.cpu().numpy().item()
        }

    def predict(self, Xi):
        Xi = check_tensor(Xi, device=self.device)
        Kid = gaussian_kernel(Xi, self.X_train, self.lengths, self.width)
        Kii = gaussian_kernel(Xi, Xi, self.lengths, self.width)
        Kdi = gaussian_kernel(self.X_train, Xi, self.lengths, self.width)

        mean = self.y_mean + (Kid @ self.Kdd_inv) @ (self.y_train - self.y_mean)
        var = Kii - Kid @ self.Kdd_inv @ Kdi

        return mean, torch.diagonal(var)**0.5

    def __repr__(self) -> str:
        hp = self.hyperparameters
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_train={self.X_train.shape[0]},\n"
            f"  n_params={self.n_params},\n"
            f"  lengths={hp['lengths'].tolist()},\n"
            f"  width={hp['width']:.4f},\n"
            f"  noise={hp['sigma']:.4f},\n"
            f"  device='{self.device}'\n"
            f")"
        )


class LGPCommittee:
    """Wrapper for a list of Local Gaussian Process models."""

    def __init__(
        self,
        lgps: list[list[LocalGaussianProcess]],
        specs: Specs,
        observations: list[int]
    ) -> None:
        """Wrapper for a list of Local Gaussian Process models."""

        if not lgps:
            raise ValueError("List of LGP models cannot be empty.")
        self.lgps = lgps
        self.observations = observations
        self.specs = specs
        self.error = None

    @property
    def slices(self) -> list[slice]:

        lengths = [lgp[0].y_size for lgp in self.lgps]
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)
        return [
            slice(start, start + length)
            for start, length in zip(offsets, lengths)
        ]

    @property
    def n_params(self) -> int:
        return self.lgps[0][0].n_params

    @property
    def committee_size(self) -> int:
        return len(self.lgps)

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict using all LGP models and return stacked means and stds."""

        # select a random models
        idx = torch.randint(0, self.committee_size, (self.committee_size, ))
        results = [com[i].predict(X) for com, i in zip(self.lgps, idx)]
        mean, var = zip(*results)

        return torch.column_stack(mean), torch.column_stack(var)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  committee_size={self.committee_size},\n"
            f"  n_params={self.n_params},\n"
            f"  n_observations={self.observations},\n"
            f")"
        )
