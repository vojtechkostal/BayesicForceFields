from pathlib import Path

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

    assert list(iter_progress(range(3), total=3, logger=logger, label="items")) == [
        0,
        1,
        2,
    ]

    out = capsys.readouterr().out
    assert "[ 33%]" in out
    assert "[100%]" in out
    assert "Done. Finished in" in out
    assert "===" not in out
