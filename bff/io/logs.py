import logging
from typing import Optional

from ..mcmc.sampler import Sampler


class Logger:
    """
    A logging utility that supports logging to a file
    or console without duplicate output.
    """

    def __init__(
        self,
        name: str,
        fn_log: Optional[str] = None,
        width: Optional[int] = 100,
        verbose: bool = True,
        mode: str = "a",
    ) -> None:

        """
        Parameters
        ----------
        name : str
            Name of the logger.
        fn_log : Optional[str], optional
            Filename to log messages to. If None, logs to console.
        width : Optional[int], default=100
            Width for message formatting.
        verbose : bool, default=True
            If False, suppresses logging output.
        mode : str, default="a"
            File open mode used when ``fn_log`` is provided.
        """

        self.fn_log = fn_log
        self.width = width
        self.verbose = verbose

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove old handlers if re-running in Jupyter
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter("%(message)s")

        if fn_log:
            handler = logging.FileHandler(fn_log, mode=mode)
        else:
            handler = logging.StreamHandler()
            handler.terminator = ""
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.propagate = False  # Prevent double logging

    def info(self, message: str, level: int, overwrite: bool = False) -> None:
        """
        Logs a message to a file or console, optionally overwriting the console output.

        Parameters
        ----------
        message : str
            The message to log.
        overwrite : bool, default=False
            If True, the message overwrites the previous console output (stdout only).
        """

        # determine width of the line automatically if not set
        width = len(message) + 10 if self.width is None else self.width

        if self.verbose:
            if level < 1:
                start = ''
            elif level == 1:
                start = '> '
            else:
                start = ' ' * 2 * (level - 1) + '- '
            message = start + f"{message}".ljust(width)
            if self.fn_log:
                self.logger.info(message)  # Always log to file if specified

            else:
                terminator = '\r' if overwrite else '\n'
                self.logger.info(message + terminator)  # Log to logger (console)


def print_progress_mcmc(
    sampler: Sampler,
    p0: np.ndarray,
    *,
    total_steps: int,
    restart: bool = True,
    fn_checkpoint: Optional[str] = None,
    logger: Optional[Logger] = None,
    **kwargs
) -> None:
    """
    Monitors and logs the convergence progress of an MCMC sampler.

    Parameters
    ----------
    sampler : Sampler
        The MCMC sampler with methods
        `sample`, `get_autocorr_time`, and attribute `iteration`.
    p0 : array-like
        Initial parameter vector for the MCMC posterior sampling.
    total_steps : int
        Total number of MCMC iterations including warmup.
    stride : int, default=100
        Frequency of autocorrelation time checks and printing.
    min_chain_length : int, default=100
        Factor to multiply the autocorrelation time with in order to
        get the minimum chain length to have proper sampling.
    rtol : float, default=0.01
        Relative tolerance of the autocorrelation time fluctuations
        for the MCMC sampling to be considered converged.
    logger : Logger, optional
        Logger instance for logging progress messages.
    **kwargs
        Additional arguments passed to the sampler's `sample` method.
    """

    logger = logger or Logger("mcmc-progress")

    rhat_tol = kwargs.get("rhat_tol", 1.01)
    ess_target = kwargs.get("ess_min", 100)

    logger.info("Posterior sampling: in progress...", level=1, overwrite=True)

    if fn_checkpoint is not None:
        logger.info(f"Checkpoint: {fn_checkpoint}", level=2)

    line = ""
    for state in sampler.run(
        p0,
        total_steps=total_steps,
        restart=restart,
        fn_checkpoint=fn_checkpoint,
        **kwargs,
    ):
        total_steps = state.total_steps
        sampling_steps = total_steps - state.warmup
        if state.phase == "warmup":
            phase_progress = f"warmup: {state.step}/{state.warmup}"
        else:
            sampling_step = max(state.step - state.warmup, 0)
            phase_progress = f"sampling: {sampling_step}/{sampling_steps}"

        if state.phase == "sampling" and state.convergence is not None:
            line = (
                f"Posterior sampling: it. {state.step}/{total_steps} | "
                f"{phase_progress} | "
                f"R-hat max: {state.convergence.max_rhat:.4f}/{rhat_tol:.4f} | "
                f"ESS min: {state.convergence.min_ess:.0f}/{ess_target:.0f}"
            )
            if state.acceptance_rate is not None:
                line += f" | acc: {state.acceptance_rate:.3f}"
            if state.it_per_sec is not None:
                line += f" | {state.it_per_sec:.0f} it/s"
        else:
            line = (
                f"Posterior sampling: it. {state.step}/{total_steps} | "
                f"{phase_progress}"
            )
            if state.acceptance_rate is not None:
                line += f" | acc: {state.acceptance_rate:.3f}"
            if state.it_per_sec is not None:
                line += f" | {state.it_per_sec:.0f} it/s"
        logger.info(line, level=1, overwrite=True)

    if line and logger.fn_log is None:
        logger.info(line, level=1)
    logger.info("", level=0)
    if sampler.converged:
        logger.info("Posterior sampling: Done.", level=1)
    else:
        logger.info(
            "Posterior sampling: Failed to converge within the maximum iterations.",
            level=1,
        )
