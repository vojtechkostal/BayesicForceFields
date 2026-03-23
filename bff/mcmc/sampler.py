import torch
import time

from pathlib import Path
from typing import Any, Optional, Callable, Tuple
from dataclasses import dataclass

from .proposal import Proposal
from .convergence import (
    ConvergenceInfo,
    integrated_autocorr_time,
    rank_normalize,
    rank_normalized_split_rhat,
)


@dataclass
class Checkpoint:
    step: int
    total_steps: int
    phase: str
    warmup: int
    thin: int
    progress_stride: int
    p: Optional[torch.Tensor] = None
    logp: Optional[torch.Tensor] = None
    accepted: Optional[torch.Tensor] = None
    posterior: Optional[torch.Tensor] = None
    acceptance_rate: Optional[float] = None
    scale: Optional[float] = None
    convergence: Optional[ConvergenceInfo] = None
    it_per_sec: Optional[float] = None
    rng_state: Optional[torch.Tensor] = None
    proposal_state: Optional[dict[str, Any]] = None
    converged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "logp": self.logp,
            "accepted": self.accepted,
            "posterior": self.posterior,
            "step": self.step,
            "total_steps": self.total_steps,
            "warmup": self.warmup,
            "thin": self.thin,
            "progress_stride": self.progress_stride,
            "rng_state": self.rng_state,
            "proposal_state": self.proposal_state,
            "converged": self.converged,
            "phase": self.phase,
            "acceptance_rate": self.acceptance_rate,
            "scale": self.scale,
            "it_per_sec": self.it_per_sec,
            "convergence": self._serialize_convergence(self.convergence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        posterior = data.get("posterior", data.get("chain"))
        if posterior is None:
            raise KeyError("Checkpoint is missing 'posterior'.")
        convergence = cls._deserialize_convergence(data.get("convergence"))
        return cls(
            p=data["p"],
            logp=data["logp"],
            accepted=data["accepted"],
            posterior=posterior,
            step=int(data["step"]),
            total_steps=int(data["total_steps"]),
            phase=data.get("phase", "sampling"),
            acceptance_rate=data.get("acceptance_rate"),
            scale=data.get("scale"),
            convergence=convergence,
            it_per_sec=data.get("it_per_sec"),
            warmup=int(data["warmup"]),
            thin=int(data["thin"]),
            progress_stride=int(data["progress_stride"]),
            rng_state=data["rng_state"],
            proposal_state=data["proposal_state"],
            converged=bool(data.get("converged", False)),
        )

    @classmethod
    def load(cls, fn: Path | str) -> "Checkpoint":
        return cls.from_dict(torch.load(fn, weights_only=False))

    def write(self, fn: Path | str) -> None:
        torch.save(self.to_dict(), fn)

    def restore(
        self,
        sampler: "Sampler",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.p is None or self.logp is None or self.accepted is None:
            raise ValueError("Checkpoint is missing sampler state.")
        if self.posterior is None:
            raise ValueError("Checkpoint is missing posterior samples.")
        if self.rng_state is None or self.proposal_state is None:
            raise ValueError("Checkpoint is missing restart metadata.")

        p = sampler._to_tensor(self.p)
        logp = sampler._to_tensor(self.logp)
        accepted = sampler._to_tensor(self.accepted).to(torch.int64)

        posterior = sampler._to_tensor(self.posterior)
        if posterior.numel():
            sampler._chain = [
                posterior[i].detach().clone() for i in range(posterior.shape[0])
            ]
        else:
            sampler._chain = []

        sampler.rng.set_state(self.rng_state)
        sampler.proposal.load_state_dict(self.proposal_state)
        sampler.converged = self.converged

        return p, logp, accepted, self.step

    @property
    def tau(self) -> torch.Tensor | None:
        if self.convergence is None or self.convergence.tau is None:
            return None
        tau = self.convergence.tau
        return tau.mean(dim=0) if tau.ndim > 1 else tau

    @staticmethod
    def _serialize_convergence(
        convergence: Optional[ConvergenceInfo],
    ) -> dict[str, torch.Tensor | None] | None:
        if convergence is None:
            return None
        return {
            "rhat": convergence.rhat.cpu() if convergence.rhat is not None else None,
            "tau": convergence.tau.cpu() if convergence.tau is not None else None,
            "ess": convergence.ess.cpu() if convergence.ess is not None else None,
        }

    @staticmethod
    def _deserialize_convergence(
        data: dict[str, torch.Tensor | None] | None,
    ) -> Optional[ConvergenceInfo]:
        if data is None:
            return None
        return ConvergenceInfo(
            rhat=data.get("rhat"),
            tau=data.get("tau"),
            ess=data.get("ess"),
        )


class Sampler:
    """
    Parallel Metropolis-Hastings sampler with walker-vectorized target evaluation.

    The sampler assumes that the target log-probability function accepts a batch
    of walker positions of shape ``(n_walkers, n_dim)`` and returns one log
    probability per walker.

    Parameters
    ----------
    log_prob : callable
        Target log-probability function with signature
        ``log_prob(x, *args) -> np.ndarray``, where ``x`` has shape
        ``(n_walkers, n_dim)`` and the return value has shape ``(n_walkers,)``.
    proposal : Proposal
        Proposal mechanism used to generate trial moves.
    args : tuple, optional
        Additional arguments passed to ``log_prob``.
    rng : np.random.Generator, optional
        Random number generator. If ``None``, a new default generator is created.
    """

    def __init__(
        self,
        log_prob: Callable,
        proposal: Proposal,
        args: Tuple = (),
        rng: Optional[torch.Generator] = None,
        device: Optional[torch.device] = 'cuda:0',
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.log_prob = log_prob
        self.proposal = proposal
        self.args = args
        self.device = device
        self.dtype = dtype

        if rng is None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(torch.seed())
        self.rng = rng

        self._chain: list[torch.Tensor] = []
        self.converged = False
        self._warmup = 0
        self._thin = 1
        self._progress_stride = 1
        self._total_steps = 0

    @property
    def chain(self) -> torch.Tensor:
        """Saved chain of walker positions.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_saved, n_walkers, n_dim)``.
            Returns an empty tensor if no samples have been saved yet.
        """
        if not self._chain:
            return torch.empty((0, 0, 0), dtype=self.dtype, device=self.device)
        return torch.stack(self._chain, dim=0)

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(
            shape, generator=self.rng, device=self.device, dtype=self.dtype)

    def _validate_inputs(
        self,
        p0: torch.Tensor,
        total_steps: int,
        warmup: int,
        thin: int,
        progress_stride: int,
    ) -> torch.Tensor:
        """Validate run inputs."""
        p = self._to_tensor(p0)

        if p.ndim != 2:
            raise ValueError("p0 must have shape (n_walkers, n_dim)")
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if warmup >= total_steps:
            raise ValueError("warmup must be smaller than total_steps")
        if thin < 1:
            raise ValueError("thin must be >= 1")
        if progress_stride < 1:
            raise ValueError("progress_stride must be >= 1")
        return p

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the target log-probability and sanitize non-finite values.

        Parameters
        ----------
        x : torch.Tensor
            Walker positions with shape ``(n_walkers, n_dim)``.

        Returns
        -------
        torch.Tensor
            Log probabilities with shape ``(n_walkers,)``. Non-finite values are
            replaced by ``-inf``.
        """
        y = self._to_tensor(self.log_prob(x, *self.args))
        if y.ndim != 1:
            raise ValueError(
                f"log_prob must return shape (n_walkers,), got {tuple(y.shape)}")
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "log_prob returned wrong length: "
                f"expected {x.shape[0]}, got {y.shape[0]}")
        return torch.where(torch.isfinite(y), y, torch.full_like(y, -torch.inf))

    def _diagnose_convergence(self, chain: torch.Tensor) -> ConvergenceInfo:
        """
        Compute convergence diagnostics from a chain tensor.
        """
        pooled = chain.reshape(-1, chain.shape[-1])
        ranked_chain = rank_normalize(pooled).reshape(chain.shape)
        rh = rank_normalized_split_rhat(chain)
        tau = integrated_autocorr_time(ranked_chain)
        n_total = chain.shape[0] * chain.shape[1]
        ess = n_total / tau.mean(dim=0)

        return ConvergenceInfo(
            rhat=rh,
            tau=tau,
            ess=ess,
        )

    def _validate_checkpoint(self, checkpoint: Checkpoint) -> None:
        if checkpoint.warmup != self._warmup:
            raise ValueError(
                f"Checkpoint warmup={checkpoint.warmup} "
                f"does not match requested warmup={self._warmup}."
            )
        if checkpoint.thin != self._thin:
            raise ValueError(
                f"Checkpoint thin={checkpoint.thin} "
                f"does not match requested thin={self._thin}."
            )

    def _should_stop(
        self,
        conv: Optional[ConvergenceInfo],
        *,
        rhat_tol: float,
        ess_min: int,
    ) -> bool:
        if conv is None:
            return False

        rhat_ok = conv.max_rhat is not None and conv.max_rhat < rhat_tol
        ess_ok = conv.min_ess is not None and conv.min_ess > ess_min
        return rhat_ok and ess_ok

    def write_posterior(self, fn: Path | str) -> None:
        torch.save({"posterior": self.chain.detach().cpu()}, fn)

    def run(
        self,
        p0: torch.Tensor,
        total_steps: int = 1500,
        warmup: int = 500,
        thin: int = 1,
        progress_stride: int = 100,
        fn_checkpoint: Optional[str | Path] = None,
        restart: bool = False,
        rhat_tol: float = 1.01,
        ess_min: int = 100,
    ):
        """
        Run the sampler.

        Parameters
        ----------
        p0 : torch.Tensor
            Initial walker positions with shape ``(n_walkers, n_dim)``.
        total_steps : int, default=1500
            Total number of MCMC steps including warmup.
        warmup : int, default=500
            Number of warmup steps during which the proposal may adapt.
        thin : int, default=1
            Save every ``thin``-th production step.
        progress_stride : int, default=100
            Frequency of yielding progress information.
        fn_chain: str or Path, optional
            If provided, the chain will be saved to this file in
            PyTorch format after each progress report.
        restart: bool, default=False
            If True and fn_chain exists, the sampler will attempt
            to load the chain from the file
            and resume sampling from the last saved position.
            Otherwise, sampling starts from p0.
        rhat_tol : float, default=1.01
            Threshold for maximum R-hat to consider the chain converged.
        ess_min : int, default=100
            Minimum effective sample size to consider the chain converged.
        Yields
        ------
        Checkpoint
            Progress snapshot for monitoring and restart.
        """

        fn_checkpoint = Path(fn_checkpoint) if fn_checkpoint is not None else None
        self.converged = False
        self._warmup = warmup
        self._thin = thin
        self._progress_stride = progress_stride
        self._total_steps = total_steps

        if restart:
            if fn_checkpoint is None or not fn_checkpoint.exists():
                raise ValueError("restart=True requires an existing checkpoint file.")
            p_probe = self._validate_inputs(
                p0, total_steps, warmup, thin, progress_stride
            )
            _, n_dim = p_probe.shape
            self.proposal.initialize(n_dim)
            checkpoint = Checkpoint.load(fn_checkpoint)
            self._validate_checkpoint(checkpoint)
            p, logp, accepted, start_step = checkpoint.restore(self)
        else:
            p = self._validate_inputs(
                p0, total_steps, warmup, thin, progress_stride
            )
            n_walkers, n_dim = p.shape
            self.proposal.initialize(n_dim)
            logp = self._log_prob(p)
            accepted = torch.zeros(n_walkers, dtype=torch.int64, device=self.device)
            self._chain = []
            start_step = 0

        n_walkers, n_dim = p.shape
        last_report_time = time.time()

        for t in range(start_step, self._total_steps):

            # propose and evaluate
            proposals = self._to_tensor(self.proposal.propose(p, self.rng))
            logp_proposed = self._log_prob(proposals)

            logu = torch.log(self._rand(n_walkers))
            accept_mask = logu < (logp_proposed - logp)

            p[accept_mask] = proposals[accept_mask]
            logp[accept_mask] = logp_proposed[accept_mask]

            # warmup / sampling
            if t < warmup:
                self.proposal.adapt(t + 1, p, accept_mask)
            else:
                accepted += accept_mask
                if (t - warmup) % thin == 0:
                    self._chain.append(p.detach().clone())

            report_progress = (
                ((t + 1) % self._progress_stride == 0)
                or (t == self._total_steps - 1)
            )
            if not report_progress:
                continue

            # number of iterations per second
            now = time.time()
            elapsed = max(now - last_report_time, 1e-12)
            steps_since_start = t + 1 - start_step if t + 1 > start_step else 1
            step_count = min(self._progress_stride, steps_since_start)
            it_per_sec = step_count / elapsed
            last_report_time = now

            phase = "warmup" if t < warmup else "sampling"
            chain = self.chain
            convergence = None
            if t >= warmup and chain.shape[0] >= 20:
                convergence = self._diagnose_convergence(chain)

            # acceptance rate during sampling phase
            acceptance_rate = None
            if t >= warmup:
                n_sampling_steps_done = t + 1 - warmup
                total_accepted = accepted.sum().item()
                acceptance_rate = total_accepted / (n_sampling_steps_done * n_walkers)

            should_stop = self._should_stop(
                convergence,
                rhat_tol=rhat_tol,
                ess_min=ess_min,
            )
            if should_stop:
                self.converged = True

            # report progress
            checkpoint = Checkpoint(
                step=t + 1,
                total_steps=self._total_steps,
                phase=phase,
                warmup=self._warmup,
                thin=self._thin,
                progress_stride=self._progress_stride,
                p=p.detach().cpu(),
                logp=logp.detach().cpu(),
                accepted=accepted.detach().cpu(),
                posterior=self.chain.detach().cpu(),
                acceptance_rate=acceptance_rate,
                scale=self.proposal.info().get("scale"),
                convergence=convergence,
                it_per_sec=it_per_sec,
                rng_state=self.rng.get_state(),
                proposal_state=self.proposal.state_dict(),
                converged=self.converged,
            )

            if fn_checkpoint is not None:
                checkpoint.write(fn_checkpoint)

            yield checkpoint

            # Early stopping based on convergence diagnostics
            if should_stop:
                break
