import logging

import time
import numpy as np
from typing import Optional

from ..mcmc.sampler import Sampler


def fmt_row(values, columns):
    return " | ".join(f"{v:^{w}}" for v, (_, w) in zip(values, columns))


def fmt_rule(columns):
    return "-+-".join("-" * w for _, w in columns)


def fmt_rule_bold(columns):
    return "===".join("=" * w for _, w in columns)


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
        verbose: bool = True
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


def print_progress(
    iterable, total: int, stride: int = 1, logger: Optional[Logger] = None
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
    sampler: Sampler,
    p0: np.ndarray,
    *,
    max_iter: int,
    restart: bool = True,
    fn_chain: Optional[str] = None,
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

    columns = [
        ("it.", 11),
        ("phase", 10),
        ("R-hat max", 12),
        ("ESS min", 10),
        ("tau CV max", 12),
        ("it/s", 8),
    ]

    conv_width = sum(w for _, w in columns[2:5]) + 2 * 3

    header_top = (
        f"{'it.':^{columns[0][1]}} | "
        f"{'phase':^{columns[1][1]}} | "
        f"{'convergence':^{conv_width}} | "
        f"{'it/s':^{columns[5][1]}}"
    )

    header_mid = fmt_row([
        "",
        "",
        "R-hat max",
        "ESS min",
        "tau CV max",
        "",
    ], columns)

    rhat_tol = kwargs.get("rhat_tol", 1.01)
    ess_target = kwargs.get("ess_min", 100)
    tau_cv_tol = kwargs.get("tau_cv_tol", 0.2)
    header_targets = fmt_row([
        f"{max_iter:.0f}",
        "-",
        f"{rhat_tol:.3g}",
        f"{ess_target:.0f}",
        f"{tau_cv_tol:.3g}",
        "-",
    ], columns)

    rule = fmt_rule(columns)
    rule_bold = fmt_rule_bold(columns)

    # Print persistent header once
    if logger is not None:
        logger.info(header_top, level=0)
        logger.info(rule, level=0)
        logger.info(header_mid, level=0)
        logger.info(header_targets, level=0)
        logger.info(rule_bold, level=0)
    last_line_len = len(rule)

    for state in sampler.run(
        p0,
        n_steps=max_iter,
        restart=restart,
        fn_chain=fn_chain,
        **kwargs,
    ):
        if state.phase == "sampling" and state.convergence is not None:
            row = [
                f"{state.step}",
                state.phase,
                f"{state.convergence.max_rhat:.6f}",
                f"{state.convergence.min_ess:.0f}",
                f"{state.convergence.tau_cv.max().item():.6f}",
                f"{state.it_per_sec:.0f}",
            ]
        else:
            row = [
                f"{state.step}",
                state.phase,
                "-",
                "-",
                "-",
                f"{state.it_per_sec:.0f}" if state.it_per_sec is not None else "-",
            ]

        line = fmt_row(row, columns).ljust(last_line_len)

        if logger is not None:
            logger.info(line, level=0, overwrite=True)

    logger.info(line, level=0)
    logger.info("", level=0)
    if sampler.converged:
        logger.info("MCMC sampling converged successfully.", level=0)
    else:
        logger.info(
            "MCMC sampling did not converge within the maximum iterations.", level=0)


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
