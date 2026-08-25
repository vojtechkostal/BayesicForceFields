from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bff.io.logs import Logger
from bff.io.progress import iter_progress
from bff.workflows._shared.campaign import run_campaign


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
    assert "[100%]" not in out
    assert "\r" not in out
    assert "Done. Finished in" in out
    assert "===" not in out


def test_status_can_be_console_only(tmp_path: Path, capsys) -> None:
    log = tmp_path / "workflow.log"
    logger = Logger("test", fn_log=log, mode="w", color=False)

    logger.status("Work", "1/2", write_file=False)

    assert "Work: 1/2" in capsys.readouterr().out
    assert log.read_text() == ""


def test_campaign_progress_is_not_written_to_physical_log(tmp_path: Path) -> None:
    log = tmp_path / "campaign.log"
    specs = tmp_path / "specs.yaml"
    specs.write_text("bounds: {}\ncharge_constraints: []\n")
    config = SimpleNamespace(
        campaign_dir=tmp_path / "campaign",
        dispatch=False,
        job_scheduler="local",
        gmx_cmd="gmx",
        store=(),
        cleanup=False,
        compress=False,
        slurm=None,
    )
    logger = Logger(
        "sample-parameters",
        fn_log=log,
        mode="w",
        verbose=False,
    )

    run_campaign(
        config=config,
        fn_specs=specs,
        systems=[],
        parameter_samples=np.array([[1.0], [2.0]]),
        logger=logger,
    )

    text = log.read_text()
    assert "Staging jobs: 1/2" not in text
    assert "Staging jobs: Done. | 2/2 [100%]" in text
