import torch

from typing import Optional


class Proposal:
    """Base class for proposal mechanisms."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

    def initialize(self, n_dim: int) -> None:
        raise NotImplementedError

    def propose(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def adapt(self, step: int, x: torch.Tensor, accepted: torch.Tensor) -> None:
        pass

    def info(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class AdaptiveGaussianProposal(Proposal):
    """
    Multivariate Gaussian random-walk proposal with optional adaptive covariance.

    The proposal has the form

    ``x_new = x + z @ L.T``,

    where ``z ~ N(0, I)`` and ``L`` is the Cholesky factor of the proposal
    covariance matrix.

    During warmup, the empirical covariance of visited states can be used to
    adapt the proposal shape and scale.

    Parameters
    ----------
    proposal_cov : torch.Tensor, optional
        Initial proposal covariance matrix of shape ``(n_dim, n_dim)``.
        If ``None``, the identity matrix is used.
    adapt : bool, default=True
        Whether to adapt the proposal during warmup.
    adapt_start : int, default=100
        First warmup step at which adaptation is allowed.
    adapt_interval : int, default=100
        Adaptation frequency in number of warmup steps.
    target_acceptance : float, default=0.234
        Target acceptance rate used for multiplicative scale adaptation.
    device : str or torch.device, default="cpu"
        Device used internally by the proposal.
    dtype : torch.dtype, default=torch.float64
        Tensor dtype used internally.

    Notes
    -----
    The covariance adaptation uses the empirical covariance of all walker
    positions seen during warmup, together with the standard scaling factor
    ``2.38**2 / n_dim``.
    """

    def __init__(
        self,
        proposal_cov: Optional[torch.Tensor] = None,
        adapt: bool = True,
        adapt_start: int = 100,
        adapt_interval: int = 100,
        target_acceptance: float = 0.234,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        self.proposal_cov = proposal_cov
        self.do_adapt = adapt
        self.adapt_start = adapt_start
        self.adapt_interval = adapt_interval
        self.target_acceptance = target_acceptance

        self.eps = 1e-6
        self.n_dim: Optional[int] = None
        self.scale = 1.0
        self.L: Optional[torch.Tensor] = None
        self.n = 0
        self.sum_x: Optional[torch.Tensor] = None
        self.sum_xx: Optional[torch.Tensor] = None

    def _to_tensor(self, x) -> torch.Tensor:
        """Convert input to the proposal device and dtype."""
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _update_cov(self, x: torch.Tensor) -> None:
        """
        Update running first and second moments from a batch of walker positions.

        Parameters
        ----------
        x : torch.Tensor
            Walker positions with shape ``(n_walkers, n_dim)``.
        """
        x = self._to_tensor(x)
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_xx += x.T @ x

    def _randn(self, shape, rng: torch.Generator) -> torch.Tensor:
        return torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)

    @property
    def cov(self) -> torch.Tensor:
        """
        Current empirical covariance estimate.

        Returns
        -------
        torch.Tensor
            Empirical covariance matrix of shape ``(n_dim, n_dim)``.
            Returns the identity matrix if fewer than two samples have been seen.
        """
        if self.n < 2:
            return torch.eye(self.n_dim, device=self.device, dtype=self.dtype)
        mean = self.sum_x / self.n
        return (self.sum_xx - self.n * torch.outer(mean, mean)) / (self.n - 1)

    def initialize(self, n_dim: int) -> None:
        """
        Initialize proposal state for a given dimensionality.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the sampled parameter vector.
        """
        self.n_dim = n_dim

        if self.proposal_cov is None:
            cov = torch.eye(n_dim, device=self.device, dtype=self.dtype)
        else:
            cov = self._to_tensor(self.proposal_cov)

        unit_matrix = torch.eye(n_dim, device=self.device, dtype=self.dtype)
        cov = 0.5 * (cov + cov.T) + self.eps * unit_matrix

        self.scale = 1.0
        self.L = torch.linalg.cholesky(cov)

        self.n = 0
        self.sum_x = torch.zeros(n_dim, device=self.device, dtype=self.dtype)
        self.sum_xx = torch.zeros((n_dim, n_dim), device=self.device, dtype=self.dtype)

    def propose(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """
        Generate Gaussian random-walk proposals.

        Parameters
        ----------
        x : torch.Tensor
            Current walker positions with shape ``(n_walkers, n_dim)``.
        rng : torch.Generator
            Random number generator.

        Returns
        -------
        torch.Tensor
            Proposed walker positions with shape ``(n_walkers, n_dim)``.

        Raises
        ------
        ValueError
            If the supplied positions have the wrong dimensionality.
        """
        x = self._to_tensor(x)
        if x.ndim != 2 or x.shape[1] != self.n_dim:
            raise ValueError(
                f"Expected shape (n_walkers, {self.n_dim}), "
                f"got {tuple(x.shape)}"
            )
        z = self._randn(x.shape, rng)
        return x + z @ self.L.T

    def adapt(self, step: int, x: torch.Tensor, accepted: torch.Tensor) -> None:
        """
        Adapt the proposal covariance and scale during warmup.

        Parameters
        ----------
        step : int
            Current warmup step number, starting from 1.
        x : torch.Tensor
            Current walker positions with shape ``(n_walkers, n_dim)``.
        accepted : torch.Tensor
            Boolean mask of shape ``(n_walkers,)`` indicating accepted walkers.
        """
        if not self.do_adapt:
            return

        self._update_cov(x)

        if step < self.adapt_start or step % self.adapt_interval != 0:
            return

        accept_mean = accepted.to(self.dtype).mean().item()
        gamma = 1.0 / step**0.5
        self.scale *= torch.exp(
            torch.tensor(
                gamma * (accept_mean - self.target_acceptance),
                device=self.device,
                dtype=self.dtype,
            )
        ).item()

        cov = (2.38**2 / self.n_dim) * self.cov
        unit_matrix = torch.eye(self.n_dim, device=self.device, dtype=self.dtype)
        cov = cov + self.eps * unit_matrix
        self.L = torch.linalg.cholesky((self.scale**2) * cov)

    def state_dict(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "scale": self.scale,
            "L": None if self.L is None else self.L.detach().cpu(),
            "n": self.n,
            "sum_x": None if self.sum_x is None else self.sum_x.detach().cpu(),
            "sum_xx": None if self.sum_xx is None else self.sum_xx.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_dim = state["n_dim"]
        self.scale = state["scale"]
        self.L = None if state["L"] is None else state["L"].to(
            self.device, self.dtype
        )
        self.n = state["n"]
        self.sum_x = None if state["sum_x"] is None else state["sum_x"].to(
            self.device, self.dtype
        )
        self.sum_xx = None if state["sum_xx"] is None else (
            state["sum_xx"].to(self.device, self.dtype)
        )
