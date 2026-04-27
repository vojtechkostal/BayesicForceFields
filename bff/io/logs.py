"""Shared logging utilities for BFF workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

    from ..mcmc.sampler import Sampler


class Logger:
    """Small workflow logger with consistent console and file output."""

    _RESET = "\033[0m"
    _STYLES = {
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "gray": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
    }
    _TITLE_STYLES = {
        "build": ("bold", "bright_cyan"),
        "reference": ("bold", "bright_blue"),
        "sample": ("bold", "bright_yellow"),
        "analyze": ("bold", "bright_magenta"),
        "fit": ("bold", "bright_green"),
        "learn": ("bold", "cyan"),
        "validate": ("bold", "bright_red"),
    }

    def __init__(
        self,
        name: str,
        fn_log: Optional[str] = None,
        width: Optional[int] = 100,
        verbose: bool = True,
        mode: str = "a",
        color: bool | str = "auto",
    ) -> None:
        self.name = name
        self.fn_log = None if fn_log is None else str(Path(fn_log).resolve())
        self.width = width
        self.verbose = verbose
        self._last_console_len = 0
        if color not in {True, False, "auto"}:
            raise ValueError("'color' must be True, False, or 'auto'.")
        self.color = sys.stdout.isatty() if color == "auto" else bool(color)

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

    def _style(self, text: str, style: str | tuple[str, ...] | None) -> str:
        if not self.color or not style or not text:
            return text
        styles = (style,) if isinstance(style, str) else style
        prefix = "".join(self._STYLES[item] for item in styles if item in self._STYLES)
        return f"{prefix}{text}{self._RESET}" if prefix else text

    def _write_console(
        self,
        line: str,
        *,
        overwrite: bool,
        style: str | tuple[str, ...] | None = None,
    ) -> None:
        if not self.verbose:
            return

        console_line = self._style(line, style)
        if overwrite:
            clear = max(self._last_console_len - len(line), 0)
            sys.stdout.write("\r" + console_line + (" " * clear))
            sys.stdout.flush()
            self._last_console_len = len(line)
            return

        self._clear_console_line()
        sys.stdout.write(console_line + "\n")
        sys.stdout.flush()

    def _write_file(self, line: str) -> None:
        if self.fn_log is None:
            return
        with Path(self.fn_log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(
        self,
        message: str = "",
        level: int = 1,
        overwrite: bool = False,
        style: str | tuple[str, ...] | None = None,
    ) -> None:
        """Write a formatted log line."""
        line = f"{self._prefix(level)}{message}" if message else ""
        self._write_file(line)
        self._write_console(line, overwrite=overwrite, style=style)

    def blank(self) -> None:
        """Write an empty line."""
        self.info("", level=0)

    def section(self, title: str) -> None:
        """Write a top-level section header."""
        title_line = f"=== {title} ==="
        self.blank()
        self.info(
            title_line,
            level=0,
            style=self._TITLE_STYLES.get(self.name, ("bold", "bright_cyan")),
        )
        self.blank()

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
        self.info(message, level=level, overwrite=overwrite, style="magenta")

    def done(
        self,
        label: str,
        *,
        detail: str | None = None,
        level: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Write a completed status line."""
        message = f"{label}: Done."
        if detail:
            message += f" | {detail}"
        self.info(message, level=level, overwrite=overwrite, style=("bold", "green"))

    def failed(
        self,
        label: str,
        *,
        detail: str | None = None,
        level: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Write a failed status line."""
        message = f"{label}: Failed."
        if detail:
            message += f" | {detail}"
        self.info(message, level=level, overwrite=overwrite, style=("bold", "red"))

    def warn(self, message: str, *, level: int = 1) -> None:
        """Write a warning line."""
        self.info(f"Warning: {message}", level=level, style="yellow")

    def right_status(
        self,
        message: str,
        status: str,
        *,
        level: int = 1,
        overwrite: bool = False,
        style: str | tuple[str, ...] | None = None,
    ) -> None:
        """Write a pytest-style line with a status block right aligned."""
        prefix = self._prefix(level)
        width = self.width or 0
        plain_len = len(prefix) + len(message) + len(status)
        spacing = " " * max(width - plain_len, 1)
        line = f"{prefix}{message}{spacing}{status}"
        self._write_file(line)
        self._write_console(line, overwrite=overwrite, style=style)

    def progress_status(
        self,
        message: str,
        current: int,
        total: int,
        *,
        level: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Write a pytest-style progress line with ``[ 42%]`` on the right."""
        percent = 100 if total <= 0 else round(100 * current / total)
        status = f"[{percent:3d}%]"
        self.right_status(
            message,
            status,
            level=level,
            overwrite=overwrite,
            style="magenta",
        )

    def result_summary(
        self,
        count: int,
        outcome: str,
        elapsed: float,
        *,
        level: int = 0,
        style: str | tuple[str, ...] | None = None,
    ) -> None:
        """Write a compact terminal completion summary."""
        if style is None:
            style = ("bold", "green") if outcome == "completed" else ("bold", "red")
        self.info(f"Done. Finished in {elapsed:.2f}s", level=level, style=style)


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
    logger.info(line, level=1, overwrite=True, style="magenta")

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
        logger.info(line, level=1, overwrite=True, style="magenta")

    logger.info(line, level=1, style="magenta")
    logger.blank()
    if sampler.converged:
        logger.done("Posterior sampling", level=1)
    else:
        logger.failed(
            "Posterior sampling",
            detail="failed to converge within the maximum iterations",
            level=1,
        )
