from pathlib import Path
from typing import Self, Union

import numpy as np
import torch

from .kernels import gaussian_kernel
from .utils import check_tensor, nearest_positive_definite, smape

PathLike = Union[str, Path]


class LocalGaussianProcess:
    """
    Local Gaussian Process Regression model.

    Parameters
    ----------
    X_train : torch.Tensor
        Training input features.
    y_train : torch.Tensor
        Training output values.
    y_mean : torch.Tensor
        Mean of the output values.
    lengths : torch.Tensor
        Length scales for the Gaussian kernel.
    width : float
        Width (amplitude) of the Gaussian kernel.
    sigma : float
        Observation noise standard deviation.
    device : str
        Device on which tensors are stored (e.g., "cpu" or "cuda").

    Attributes
    ----------
    Kdd_inv : torch.Tensor
        Inverse of the training covariance matrix.

    Methods
    -------
    predict(Xi: torch.Tensor) -> torch.Tensor
        Predict outputs for given input points.

    Properties
    ----------
    n_params : int
        Number of input parameters (features).
    y_size : int
        Dimensionality of the output values.
    hyperparameters : Dict[str, Union[np.ndarray, float]]
        Dictionary of hyperparameters used in the model.

    """

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
        Kdd = nearest_positive_definite(Kdd)
        L = torch.linalg.cholesky(Kdd)
        Kdd_inv = torch.cholesky_inverse(L)
        self.Kdd_inv = check_tensor(Kdd_inv, device=device)

    @property
    def n_params(self) -> int:
        return self.X_train.shape[1]

    @property
    def y_size(self) -> int:
        return self.y_train.shape[1]

    @property
    def hyperparameters(self) -> dict[str, Union[np.ndarray, float]]:
        return {
            'lengths': self.lengths.cpu().numpy(),
            'width': self.width.cpu().numpy().item(),
            'sigma': self.sigma.cpu().numpy().item()
        }

    @torch.no_grad()
    def predict(self, Xi: torch.Tensor) -> torch.Tensor:
        """
        Predict outputs for given input points.

        Parameters
        ----------
        Xi : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Predicted outputs of shape (n_samples, output_dim).
        """
        Xi = check_tensor(Xi, device=self.device)
        Kid = gaussian_kernel(Xi, self.X_train, self.lengths, self.width)
        mean = self.y_mean + (Kid @ self.Kdd_inv) @ (self.y_train - self.y_mean)
        return mean

    def state_dict(self) -> dict:
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "y_mean": self.y_mean,
            "lengths": self.lengths,
            "width": self.width,
            "sigma": self.sigma,
            "device": self.device,
        }

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
    """
    Committee of Local Gaussian Process (LGP) models.

    Parameters
    ----------
    lgps : list of LocalGaussianProcess
        List of LGP models.
    n_observations : int
        Effective number of observations used in the likelihood term.
    reference_values : np.ndarray
        Reference observation vector matched by the surrogate outputs.
    nuisance : float, optional
        Nuisance parameter for model selection (default is None).
    stochastic : bool, optional
        If True, randomly select one model for prediction instead of averaging
        (default is False).

    Properties
    ----------
    size : int
        Number of LGP models in the committee.
    n_params : int
        Number of input parameters (features) used by the LGP models.

    Methods
    -------
    predict(X: torch.Tensor) -> torch.Tensor
        Predict outputs by averaging predictions from all LGP models
        or randomly selecting one if stochastic is True.
    validate(X_test: torch.Tensor, y_test: torch.Tensor) -> float
        Validate the committee by computing the
        mean squared error of predictions against test data.
    """
    def __init__(
        self,
        lgps: list[LocalGaussianProcess],
        n_observations: int,
        reference_values: np.ndarray,
        nuisance: float | None = None,
        stochastic: bool = False
    ) -> None:
        self.lgps = lgps
        self.error: float | None = None
        self.n_observations = int(n_observations)
        self.reference_values = np.asarray(reference_values, dtype=float).reshape(-1)
        self.nuisance = nuisance
        self.stochastic = stochastic

        if self.reference_values.size != self.lgps[0].y_size:
            raise ValueError(
                "Reference observation size does not match surrogate output size."
            )

    @property
    def size(self) -> int:
        return len(self.lgps)

    @property
    def n_params(self) -> int:
        return self.lgps[0].n_params

    @property
    def y_size(self) -> int:
        return self.lgps[0].y_size

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict outputs by averaging predictions from all LGP models.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Averaged predicted outputs of shape (n_samples, output_dim).
        """

        if self.size > 1 and self.stochastic:
            # Select a random model and return its prediction
            idx = torch.randint(self.size, (1,)).item()
            return self.lgps[idx].predict(X)

        # Otherwise, return the mean prediction across all models
        predictions = torch.stack([lgp.predict(X) for lgp in self.lgps])
        return predictions.mean(dim=0).squeeze(0)

    def validate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """
        Validate the committee by computing the mean squared error.

        Parameters
        ----------
        X_test : torch.Tensor
            Test input tensor of shape (n_samples, n_features).
        y_test : torch.Tensor
            Test output tensor of shape (n_samples, output_dim).

        Returns
        -------
        float
            Mean squared error of the predictions.
        """

        y_pred = self.predict(X_test)
        self.error = 100.0 * smape(y_test, y_pred)
        return self.error

    @classmethod
    def load(cls, fn: PathLike) -> Self:
        state = torch.load(fn, weights_only=False)
        lgps = [LocalGaussianProcess(**lgp_state) for lgp_state in state["lgps"]]
        committee = cls(
            lgps=lgps,
            n_observations=state["n_observations"],
            reference_values=np.asarray(state["reference_values"], dtype=float),
            nuisance=state["nuisance"],
            stochastic=state["stochastic"],
        )
        committee.error = state["error"]
        return committee

    def write(self, fn_out: PathLike) -> None:

        state = {
            "n_observations": self.n_observations,
            "reference_values": self.reference_values.tolist(),
            "nuisance": self.nuisance,
            "stochastic": self.stochastic,
            "error": self.error,
            "lgps": [lgp.state_dict() for lgp in self.lgps],
        }

        torch.save(state, fn_out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  committee_size={self.size},\n"
            f"  n_params={self.n_params},\n"
            f"  testset error={self.error},\n"
            f")"
        )
