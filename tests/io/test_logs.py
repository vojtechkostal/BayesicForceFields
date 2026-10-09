import io
import sys
from pathlib import Path

import pytest

from bff.io.logs import Logger
from bff.io.progress import iter_progress


def test_logger_writes_colored_console_and_plain_file(
    tmp_path: Path,
    capsys,
) -> None:
    log = tmp_path / "workflow.log"
    logger = Logger("test", fn_log=log, mode="w", color=True, width=40)

    logger.done("Step", detail="1/1")

    console = capsys.readouterr().out
    file_text = log.read_text()

    assert "\033[32m" in console
    assert "Step: Done. | 1/1" in console
    assert "\033[" not in file_text
    assert "Step: Done. | 1/1" in file_text


def test_logger_progress_status_right_aligns_percentage(capsys) -> None:
    logger = Logger("test", color=False, width=40)

    logger.progress_status("tests/bayes/test_file.py ....", 4, 10)

    line = capsys.readouterr().out.rstrip("\n")
    assert line.endswith("[ 40%]")
    assert len(line) == 40


def test_iter_progress_prints_pytest_style_summary(capsys) -> None:
    logger = Logger("test", color=False, width=50)

    items = iter_progress(range(3), total=3, logger=logger, label="items", log_every=1)
    assert list(items) == [0, 1, 2]

    out = capsys.readouterr().out
    assert "[ 33%]" in out
    assert "[100%]" not in out
    assert "items: Done. | 3/3 in 0s" in out
    assert "===" not in out


def test_iter_progress_logs_every_n_items_and_the_console_follows(
    tmp_path: Path, capsys
) -> None:
    log = tmp_path / "workflow.log"
    logger = Logger("test", fn_log=log, mode="w", color=False)

    list(iter_progress(range(250), total=250, logger=logger, label="items"))

    lines = log.read_text().splitlines()
    assert [line.split("|")[0].strip() for line in lines[:2]] == [
        "> items: 100/250",
        "> items: 200/250",
    ]
    assert "items: Done. | 250/250" in lines[2]
    # Not a terminal: the console shows the same lines, not one per item.
    assert capsys.readouterr().out.count("items:") == 3


def test_iter_progress_rejects_a_bad_stride() -> None:
    with pytest.raises(ValueError, match="log_every"):
        list(
            iter_progress(
                range(3), total=3, logger=Logger("t"), label="x", log_every=0
            )
        )


def test_status_can_be_console_only(tmp_path: Path, capsys) -> None:
    log = tmp_path / "workflow.log"
    logger = Logger("test", fn_log=log, mode="w", color=False)

    logger.status("Work", "1/2", write_file=False)

    assert "Work: 1/2" in capsys.readouterr().out
    assert log.read_text() == ""


def test_non_tty_progress_does_not_overwrite_lines(monkeypatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    logger = Logger("sample")
    logger.status("Work", "1/2", overwrite=True)
    logger.done("Work")
    assert "\r" not in stream.getvalue()
