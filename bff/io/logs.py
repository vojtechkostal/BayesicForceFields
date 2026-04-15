"""Shared logging utilities for BFF workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ..mcmc.sampler import Sampler


class Logger:
    """Small workflow logger with consistent console and file output."""

    def __init__(
        self,
        name: str,
        fn_log: Optional[str] = None,
        width: Optional[int] = 100,
        verbose: bool = True,
        mode: str = "a",
    ) -> None:
        self.name = name
        self.fn_log = None if fn_log is None else str(Path(fn_log).resolve())
        self.width = width
        self.verbose = verbose
        self._last_console_len = 0

        if mode not in {"a", "w"}:
            raise ValueError("'mode' must be either 'a' or 'w'.")

        if self.fn_log is not None:
            log_path = Path(self.fn_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "w":
                log_path.write_text("", encoding="utf-8")
            elif not log_path.exists():
                log_path.touch()

    def _prefix(self, level: int) -> str:
        if level <= 0:
            return ""
        if level == 1:
            return "> "
        return "  " * (level - 1) + "- "

    def _clear_console_line(self) -> None:
        if self._last_console_len == 0:
            return
        sys.stdout.write("\r" + (" " * self._last_console_len) + "\r")
        sys.stdout.flush()
        self._last_console_len = 0

    def _write_console(self, line: str, *, overwrite: bool) -> None:
        if not self.verbose:
            return

        if overwrite:
            clear = max(self._last_console_len - len(line), 0)
            sys.stdout.write("\r" + line + (" " * clear))
            sys.stdout.flush()
            self._last_console_len = len(line)
            return

        self._clear_console_line()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _write_file(self, line: str) -> None:
        if self.fn_log is None:
            return
        with Path(self.fn_log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(self, message: str = "", level: int = 1, overwrite: bool = False) -> None:
        """Write a formatted log line."""
        line = f"{self._prefix(level)}{message}" if message else ""
        self._write_file(line)
        self._write_console(line, overwrite=overwrite)

    def blank(self) -> None:
        """Write an empty line."""
        self.info("", level=0)

    def section(self, title: str) -> None:
        """Write a top-level section header."""
        self.info(title, level=0)
        self.info("=" * len(title), level=0)

    def kv(self, key: str, value: object, *, level: int = 1) -> None:
        """Write one key-value summary line."""
        self.info(f"{key}: {value}", level=level)

    def status(
        self,
        label: str,
        state: str,
        *,
        detail: str | None = None,
        level: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Write one workflow status line."""
        message = f"{label}: {state}"
        if detail:
            message += f" | {detail}"
        self.info(message, level=level, overwrite=overwrite)

    def done(
        self,
        label: str,
        *,
        detail: str | None = None,
        level: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Write a completed status line."""
        self.status(
            label,
            "Done.",
            detail=detail,
            level=level,
            overwrite=overwrite,
        )

    def warn(self, message: str, *, level: int = 1) -> None:
        """Write a warning line."""
        self.info(f"Warning: {message}", level=level)

    def warn_if(self, condition: bool, message: str, *, level: int = 1) -> None:
        """Write a warning line when ``condition`` is true."""
        if condition:
            self.warn(message, level=level)


def print_progress_mcmc(
    sampler: Sampler,
    p0: np.ndarray,
    *,
    total_steps: int,
    restart: bool = True,
    fn_checkpoint: Optional[str] = None,
    logger: Optional[Logger] = None,
    **kwargs,
) -> None:
    """Monitor and log MCMC sampling progress."""
    logger = logger or Logger("mcmc-progress")

    rhat_tol = kwargs.get("rhat_tol", 1.01)
    ess_target = kwargs.get("ess_min", 100)
    total_digits = len(str(total_steps))
    warmup_digits = len(str(kwargs.get("warmup", total_steps)))
    sampling_digits = len(str(max(total_steps - kwargs.get("warmup", 0), 0)))
    ess_digits = max(len(str(int(ess_target))), 1)

    def _fmt_progress(current: int, total: int, width: int, label: str) -> str:
        return f"{label}: {current:>{width}d}/{total:<{width}d}"

    if fn_checkpoint is not None:
        logger.kv("Checkpoint", fn_checkpoint, level=2)

    line = (
        f"Posterior sampling: it. {0:>{total_digits}d}/{total_steps:<{total_digits}d}"
    )
    logger.info(line, level=1, overwrite=True)

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
            phase_progress = _fmt_progress(
                state.step,
                state.warmup,
                warmup_digits,
                "warmup",
            )
        else:
            sampling_step = max(state.step - state.warmup, 0)
            phase_progress = _fmt_progress(
                sampling_step,
                sampling_steps,
                sampling_digits,
                "sampling",
            )

        if state.phase == "sampling" and state.convergence is not None:
            line = (
                f"Posterior sampling: it. "
                f"{state.step:>{total_digits}d}/{total_steps:<{total_digits}d} | "
                f"{phase_progress} | "
                f"R-hat max: {state.convergence.max_rhat:.4f}/{rhat_tol:.4f} | "
                f"ESS min: {state.convergence.min_ess:>{ess_digits}.0f}/"
                f"{ess_target:<{ess_digits}.0f}"
            )
            if state.acceptance_rate is not None:
                line += f" | acc: {state.acceptance_rate:.3f}"
            if state.it_per_sec is not None:
                line += f" | {state.it_per_sec:>3.0f} it/s"
        else:
            line = (
                f"Posterior sampling: it. "
                f"{state.step:>{total_digits}d}/{total_steps:<{total_digits}d} | "
                f"{phase_progress}"
            )
            if state.acceptance_rate is not None:
                line += f" | acc: {state.acceptance_rate:.3f}"
            if state.it_per_sec is not None:
                line += f" | {state.it_per_sec:>3.0f} it/s"
        logger.info(line, level=1, overwrite=True)

    logger.info(line, level=1)
    logger.blank()
    if sampler.converged:
        logger.done("Posterior sampling", level=1)
    else:
        logger.warn(
            "Posterior sampling failed to converge within the maximum iterations.",
            level=1,
        )
