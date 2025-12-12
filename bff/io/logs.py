import logging

import time
import numpy as np


class Logger:
    """
    A logging utility that supports logging to a file
    or console without duplicate output.
    """

    def __init__(
        self, name: str, fn_log: str = None, width: int = None, verbose: bool = True
    ) -> None:

        """
        Parameters
        ----------
        name : str
            Name of the logger.
        fn_log : str, optional
            Filename to log messages to. If None, logs to console.
        width : int, default=None
            Width for message formatting.
        verbose : bool, default=True
            If False, suppresses logging output.
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
            handler = logging.FileHandler(fn_log)
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
        if self.width is None:
            width = len(message) + 10

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
    pad = len(str(total))
    logger = logger or Logger('progress')

    progress_template = (
        "Training QoI: {i:>{pad}}/{total} "
        "({percent:3.0f}%) | {elapsed} < {eta}"
    )

    progress_str = progress_template.format(
        i=0, pad=pad, total=total, percent=0, elapsed='0s', eta='NaN'
    )

    logger.info(progress_str, level=1, overwrite=True)

    for i, item in enumerate(iterable, 1):
        yield item

        if i % stride == 0:
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / i
            eta = time_per_step * (total - i)

            progress_str = progress_template.format(
                i=i, pad=pad, total=total,
                percent=100 * i / total,
                elapsed=format_time(elapsed_time),
                eta=format_time(eta)
            )

            logger.info(progress_str, level=1, overwrite=True)

        i += 1


def print_progress_mcmc(
    sampler,
    p0: np.ndarray,
    max_iter: int,
    stride: int = 100,
    min_chain_length: int = 100,
    rtol: float = 0.01,
    logger: Logger = None,
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
    **kwargs
        Additional arguments passed to the sampler's `sample` method.
    """

    t0 = time.time()
    tau = np.inf
    generator = sampler.sample(
        p0, iterations=max_iter, progress=False, store=True, **kwargs
    )
    pad = len(str(max_iter))
    logger = logger or Logger('mcmc')

    for i, sample in enumerate(generator, start=1):
        it = sampler.iteration
        if it >= max_iter:
            logger.info(f"MCMC did not converge in {it} iterations.", level=2)
            break

        if i == 1:
            progress_str = f"it. {0:>{pad}}/{max_iter}".rjust(pad)
            logger.info(progress_str, level=1, overwrite=True)

        # Check convergence every `stride` steps
        if it % stride == 0:
            converged, progress_chain, progress_fluct, tau_new = mcmc_convergence(
                sampler, it, tau, min_chain_length, rtol
            )
            tau = tau_new
            elapsed_time = time.time() - t0
            it_per_sec = int((i + 1) / elapsed_time) if elapsed_time > 0 else 0

            logger.info(
                f"it. {it:>{pad}}/{max_iter}".rjust(pad) + " | "
                f"{it_per_sec} it/s | "
                "convergence: "
                f"length = {100 * progress_chain:3.0f}%, "
                f"fluctuations = {100 * progress_fluct:3.0f}%",
                level=1,
                overwrite=True
            )

            if converged:
                t1 = time.time()
                logger.info(
                    f"MCMC converged in {it} it. & {format_time(t1 - t0)}", level=1
                )
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


def format_time(seconds: float) -> str:
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
