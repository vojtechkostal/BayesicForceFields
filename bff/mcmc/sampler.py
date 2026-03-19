import torch
import time

from pathlib import Path
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from .proposal import Proposal
from .convergence import (
    ConvergenceInfo,
    split_rhat,
    stability_metrics,
    integrated_autocorr_time
)


@dataclass
class StateInfo:
    step: int
    total_steps: int
    phase: str
    acceptance_rate: Optional[float] = None
    scale: Optional[float] = None
    convergence: Optional[ConvergenceInfo] = None
    it_per_sec: Optional[float] = None


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
        n_steps: int,
        warmup: int,
        thin: int,
        progress_stride: int,
    ) -> torch.Tensor:
        """Validate run inputs."""
        p = self._to_tensor(p0)

        if p.ndim != 2:
            raise ValueError("p0 must have shape (n_walkers, n_dim)")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
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

        rh = split_rhat(chain)
        mean_rel_change, std_rel_change = stability_metrics(chain)
        tau = integrated_autocorr_time(chain)
        n_total = chain.shape[0] * chain.shape[1]
        ess = n_total / tau.mean(dim=0)

        return ConvergenceInfo(
            rhat=rh,
            tau=tau,
            ess=ess,
            mean_rel_change=mean_rel_change,
            std_rel_change=std_rel_change,
        )

    def _write_checkpoint(
        self,
        fn: Path | None,
        *,
        p: torch.Tensor,
        logp: torch.Tensor,
        accepted: torch.Tensor,
        step: int,
        total_steps: int,
        warmup: int,
        thin: int,
        progress_stride: int,
        state,
    ) -> None:
        """Write restartable checkpoint."""
        if fn is None:
            return

        checkpoint_dict = {
            "p": p.detach().cpu(),
            "logp": logp.detach().cpu(),
            "accepted": accepted.detach().cpu(),
            "chain": self.chain.detach().cpu(),
            "step": step,
            "total_steps": total_steps,
            "warmup": warmup,
            "thin": thin,
            "progress_stride": progress_stride,
            "rng_state": self.rng.get_state(),
            "proposal_state": self.proposal.state_dict(),
            "converged": self.converged,
            "state": {
                "step": state.step,
                "total_steps": state.total_steps,
                "phase": state.phase,
                "acceptance_rate": state.acceptance_rate,
                "scale": state.scale,
                "it_per_sec": state.it_per_sec,
                "convergence": (
                    {
                        "rhat": (
                            state.convergence.rhat.cpu()
                            if state.convergence
                            and state.convergence.rhat is not None
                            else None
                        ),
                        "tau": (
                            state.convergence.tau.cpu()
                            if state.convergence
                            and state.convergence.tau is not None
                            else None
                        ),
                        "ess": (
                            state.convergence.ess.cpu()
                            if state.convergence
                            and state.convergence.ess is not None
                            else None
                        ),
                        "mean_rel_change": (
                            state.convergence.mean_rel_change.cpu()
                            if state.convergence
                            and state.convergence.mean_rel_change is not None
                            else None
                        ),
                        "std_rel_change": (
                            state.convergence.std_rel_change.cpu()
                            if state.convergence
                            and state.convergence.std_rel_change is not None
                            else None
                        ),
                    }
                    if state.convergence
                    else None
                ),
            }
        }

        torch.save(checkpoint_dict, fn)

    def _load_checkpoint(
        self,
        fn: Path,
        *,
        warmup: int,
        thin: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        ckpt = torch.load(fn, weights_only=False)

        if ckpt["warmup"] != warmup:
            raise ValueError(
                f"Checkpoint warmup={ckpt['warmup']} "
                f"does not match requested warmup={warmup}.")
        if ckpt["thin"] != thin:
            raise ValueError(
                f"Checkpoint thin={ckpt['thin']} "
                f"does not match requested thin={thin}.")

        p = self._to_tensor(ckpt["p"])
        logp = self._to_tensor(ckpt["logp"])
        accepted = self._to_tensor(ckpt["accepted"]).to(torch.int64)

        chain = self._to_tensor(ckpt["chain"])
        if chain.numel():
            self._chain = [chain[i].detach().clone() for i in range(chain.shape[0])]
        else:
            self._chain = []

        self.rng.set_state(ckpt["rng_state"])
        self.proposal.load_state_dict(ckpt["proposal_state"])
        self.converged = bool(ckpt.get("converged", False))

        start_step = int(ckpt["step"])
        if start_step >= total_steps:
            raise ValueError(
                f"Checkpoint step={start_step} "
                f"already reached total_steps={total_steps}.")

        return p, logp, accepted, start_step

    def _should_stop(
        self,
        conv: Optional[ConvergenceInfo],
        *,
        rhat_tol: float,
        ess_min: int,
        tau_cv_max: float,
    ) -> bool:
        if conv is None:
            return False

        rhat_ok = conv.max_rhat is not None and conv.max_rhat < rhat_tol
        ess_ok = conv.min_ess is not None and conv.min_ess > ess_min
        tau_cv_ok = (
            conv.tau_cv is not None
            and torch.all(conv.tau_cv < tau_cv_max).item()
        )

        return rhat_ok and ess_ok and tau_cv_ok

    def run(
        self,
        p0: torch.Tensor,
        n_steps: int = 1000,
        warmup: int = 500,
        thin: int = 1,
        progress_stride: int = 100,
        fn_chain: Optional[str | Path] = None,
        restart: bool = False,
        rhat_tol: float = 1.01,
        ess_min: int = 100,
        tau_cv_max: float = 0.2,
    ):
        """
        Run the sampler.

        Parameters
        ----------
        p0 : torch.Tensor
            Initial walker positions with shape ``(n_walkers, n_dim)``.
        n_steps : int, default=1000
            Number of production steps.
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
        tau_cv_max : float, default=0.2
            Maximum coefficient of variation of autocorrelation times across parameters

        Yields
        ------
        StateInfo
            Progrss snapshot for monitoring
        """

        fn_chain = Path(fn_chain) if fn_chain is not None else None
        self.converged = False

        total_steps = warmup + n_steps

        if restart:
            if fn_chain is None or not fn_chain.exists():
                raise ValueError("restart=True requires an existing checkpoint file.")
            p_probe = self._validate_inputs(p0, n_steps, warmup, thin, progress_stride)
            _, n_dim = p_probe.shape
            self.proposal.initialize(n_dim)
            p, logp, accepted, start_step = self._load_checkpoint(
                fn_chain,
                warmup=warmup,
                thin=thin,
                total_steps=total_steps,
            )
        else:
            p = self._validate_inputs(p0, n_steps, warmup, thin, progress_stride)
            n_walkers, n_dim = p.shape
            self.proposal.initialize(n_dim)
            logp = self._log_prob(p)
            accepted = torch.zeros(n_walkers, dtype=torch.int64, device=self.device)
            self._chain = []
            start_step = 0

        n_walkers, n_dim = p.shape
        last_report_time = time.time()

        for t in range(start_step, total_steps):

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

            report_progress = ((t + 1) % progress_stride == 0) or (t == total_steps - 1)
            if not report_progress:
                continue

            # number of iterations per second
            now = time.time()
            elapsed = max(now - last_report_time, 1e-12)
            steps_since_start = t + 1 - start_step if t + 1 > start_step else 1
            step_count = min(progress_stride, steps_since_start)
            it_per_sec = step_count / elapsed
            last_report_time = now

            phase = "warmup" if t < warmup else "sampling"
            convergence = self._diagnose_convergence(self.chain) if t > warmup else None

            # acceptance rate during sampling phase
            acceptance_rate = None
            if t >= warmup:
                n_sampling_steps_done = t + 1 - warmup
                total_accepted = accepted.sum().item()
                acceptance_rate = total_accepted / (n_sampling_steps_done * n_walkers)

            # report progress
            state = StateInfo(
                step=t + 1,
                total_steps=total_steps,
                phase=phase,
                acceptance_rate=acceptance_rate,
                scale=self.proposal.info().get("scale"),
                convergence=convergence,
                it_per_sec=it_per_sec,
            )

            self._write_checkpoint(
                fn_chain,
                p=p,
                logp=logp,
                accepted=accepted,
                step=t + 1,
                total_steps=total_steps,
                warmup=warmup,
                thin=thin,
                state=state,
                progress_stride=progress_stride,
            )

            yield state

            # Early stopping based on convergence diagnostics
            if self._should_stop(
                convergence,
                rhat_tol=rhat_tol,
                ess_min=ess_min,
                tau_cv_max=tau_cv_max,
            ):
                self.converged = True
                break
