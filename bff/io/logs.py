import logging

import time
import numpy as np


class Logger:
    """
    A logging utility that supports logging to a file
    or console without duplicate output.

    Parameters
    ----------
    fn_log : str, optional
        Path to the log file. If None, logs are printed to the console.
    width : int, default=100
        Ensures full overwriting of progress messages in the console.
    verbose : bool, default=True
        Whether to print messages to stdout when logging to file.
    """

    def __init__(
        self, fn_log: str = None, width: int = 100, verbose: bool = True
    ) -> None:
        self.fn_log = fn_log
        self.width = width
        self.verbose = verbose
        self.logger = logging.getLogger("custom_logger")
        self.logger.setLevel(logging.INFO)

        # Remove old handlers if re-running in Jupyter
        self.logger.handlers.clear()

        if self.fn_log:
            # Only add file handler if logging to file
            handler = logging.FileHandler(self.fn_log)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def info(self, message: str, overwrite: bool = False) -> None:
        """
        Logs a message to a file or console, optionally overwriting the console output.

        Parameters
        ----------
        message : str
            The message to log.
        overwrite : bool, default=False
            If True, the message overwrites the previous console output (stdout only).
        """
        if self.fn_log:
            self.logger.info(message)  # Always log to file if specified

        # Handle stdout printing separately
        if self.verbose and self.fn_log is None:
            if overwrite:
                print(message.ljust(self.width), end='\r', flush=True)
            else:
                print(message.ljust(self.width), flush=True)


def print_progress(
    iterable, total: int, stride: int = 1, logger: Logger = None
):
    """
    Prints and logs progress of an iterable process,
    including elapsed time and estimated time remaining.

    Parameters
    ----------
    iterable : iterable
        The iterable to process.
    total : int
        The total number of items in the iterable.
    stride : int, default=1
        How frequently progress should be logged.
    logger : Logger, optional
        Logger instance for logging progress messages.
    """
    start_time = time.time()
    pad = len(str(total)) + 1

    for i, item in enumerate(iterable, 1):
        yield item

        if i % stride == 0:
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / i
            eta = time_per_step * (total - i)

            progress_str = (
                f"> loading training trajectories: {i:>{pad}}/{total} "
                f"({100 * i / total:3.0f}%) | "
                f"{format_time(elapsed_time)} < {format_time(eta)}"
            )

            if logger is not None:
                logger.info(progress_str, overwrite=True)

        i += 1


def print_progress_mcmc(
    sampler,
    p0: np.ndarray,
    max_iter: int,
    stride: int = 100,
    min_chain_length: int = 100,
    rtol: float = 0.01,
    logger: Logger = None,
    title: str = 'MCMC',
    **kwargs
) -> None:
    """
    Monitors and logs the convergence progress of an MCMC sampler.

    Parameters
    ----------
    sampler : object
        The MCMC sampler with methods
        `sample`, `get_autocorr_time`, and attribute `iteration`.
    p0 : array-like
        Initial parameter vector for the MCMC chain.
    max_iter : int
        Maximum number of iterations.
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
    title : str, default='MCMC'
        Title for the progress messages.
    **kwargs
        Additional arguments passed to the sampler's `sample` method.
    """

    t0 = time.time()
    tau = np.inf
    generator = sampler.sample(
        p0, iterations=max_iter, progress=False, store=True, **kwargs
    )
    pad = len(str(max_iter)) + 1

    for i, sample in enumerate(generator, start=1):
        it = sampler.iteration
        if it >= max_iter:
            logger.info(f"> {title}: Did not converge in {it} iterations.")
            break

        if i == 1:
            progress_str = f"> {title}: {0:>{pad}}/{max_iter}".rjust(pad)
            logger.info(progress_str, overwrite=True)

        # Check convergence every `stride` steps
        if it % stride == 0:
            converged, progress_chain, progress_fluct, tau_new = mcmc_convergence(
                sampler, it, tau, min_chain_length, rtol
            )
            tau = tau_new
            elapsed_time = time.time() - t0
            it_per_sec = int((i + 1) / elapsed_time) if elapsed_time > 0 else 0

            logger.info(
                f"> {title}: {it:>{pad}}/{max_iter}".rjust(pad) + ' | '
                f"{it_per_sec} it/s | "
                f"chain: {100 * progress_chain:3.0f}%, "
                f"fluct: {100 * progress_fluct:3.0f}%",
                overwrite=True
            )

            if converged:
                t1 = time.time()
                logger.info(f"> {title}: Done. ({it} it. & {format_time(t1 - t0)})")
                break


def mcmc_convergence(
        sampler, it: int, tau: np.ndarray, min_chain_length: int, rtol: float
) -> tuple[bool, float, float, np.ndarray]:
    """Check convergence of MCMC sampling based on autocorrelation times."""

    tau_new = sampler.get_autocorr_time(tol=0)
    converged = False

    # Calculate convergence criteria
    crit_chain = tau_new * min_chain_length
    crit_fluct = (
        np.abs(tau - tau_new) / tau_new
        if np.any(tau_new > 0)
        else np.inf
    )
    progress_chain = np.clip(it / crit_chain.max(), 0, 1)
    progress_fluct = np.clip(rtol / np.max(crit_fluct), 0, 1)

    if np.all(crit_chain < it) and np.all(crit_fluct < rtol):
        converged = True

    return converged, progress_chain, progress_fluct, tau_new


def format_time(seconds):
    """Format seconds as hours, minutes, and seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif (hours == 0) & (minutes > 0):
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = f"{int(seconds)}s"

    return time_str
