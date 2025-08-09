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
    def __init__(self, lgps: list[LocalGaussianProcess], n_observations: int) -> None:
        self.lgps = lgps
        self.error = None
        self.observations = n_observations

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
