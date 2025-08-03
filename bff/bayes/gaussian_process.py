import torch
from .kernels import gaussian_kernel
from .utils import check_tensor, nearest_positive_definite
from ..evaluation.metrics import mape_fn


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
    """
    def __init__(self, lgps: list[LocalGaussianProcess]) -> None:
        self.lgps = lgps
        self.error = None

    @property
    def size(self):
        return len(self.lgps)

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
        return torch.stack([lgp.predict(X) for lgp in self.lgps]).mean(dim=0)

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

        y_pred = self.predict(X_test).cpu().numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.cpu().numpy()
        self.error = mape_fn(y_test, y_pred) * 100

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  committee_size={self.size},\n"
            f"  n_params={self.lgps[0].n_params},\n"
            f"  testset error={self.error},\n"
            f")"
        )


class CommitteeWrapper:
    """
    Wrapper for a list of Local Gaussian Process committees.

    Parameters
    ----------
    committees : list of LGPCommittee
        List of committees, each handling a subset of the output.
    observations : list of int
        List of indices representing which outputs are observed.
    """
    def __init__(
        self,
        committees: list[LGPCommittee],
        observations: list[int],
    ) -> None:
        """Wrapper for a list of Local Gaussian Process models."""

        self.committees = committees
        self.observations = observations

    @property
    def n_params(self) -> int:
        n = [lgp.n_params for com in self.committees for lgp in com.lgps]
        if len(set(n)) != 1:
            raise ValueError("All LGPs must have the same number of parameters.")
        return n[0]

    @property
    def slices(self) -> list[slice]:

        lengths = [com.lgps[0].y_size for com in self.committees]
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)
        return [
            slice(start, start + length)
            for start, length in zip(offsets, lengths)
        ]

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict outputs for all output dimensions using all LGP committees.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        tuple of torch.Tensor
            Stacked predictions of shape (n_samples, total_output_dim).
        """
        return torch.column_stack([com.predict(X) for com in self.committees])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_params={self.n_params},\n"
            f"  n_observations={self.observations},\n"
            f")"
        )
