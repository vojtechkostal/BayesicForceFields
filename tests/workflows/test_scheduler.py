from types import SimpleNamespace

from bff.workflows._shared import scheduler


def test_get_job_state_counts_splits_slurm_states(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="101,PD\n102,R\n103,CG\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)

    counts = scheduler.get_job_state_counts([101, 102, 103, 104], "slurm")

    assert counts == {
        "submitted": 4,
        "pending": 1,
        "running": 2,
        "finished": 1,
        "active": 3,
        "unknown": 0,
    }


def test_get_job_state_counts_includes_last_line_without_newline(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="101,R\n102,R",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)

    counts = scheduler.get_job_state_counts([101, 102], "slurm")

    assert counts["active"] == 2
    assert counts["running"] == 2
    assert counts["finished"] == 0


def test_wait_for_scheduler_slot_reports_counts(monkeypatch) -> None:
    calls = [
        {
            "submitted": 2,
            "pending": 1,
            "running": 1,
            "finished": 0,
            "active": 2,
            "unknown": 0,
        },
        {
            "submitted": 2,
            "pending": 0,
            "running": 1,
            "finished": 1,
            "active": 1,
            "unknown": 0,
        },
    ]
    reported = []

    monkeypatch.setattr(
        scheduler,
        "get_job_state_counts",
        lambda ids, sched: calls.pop(0),
    )
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)

    scheduler.wait_for_scheduler_slot(
        job_ids=[1, 2],
        scheduler="slurm",
        max_parallel_jobs=2,
        monitor=reported.append,
        poll_interval=0,
    )

    assert reported[0]["active"] == 2
    assert reported[-1]["active"] == 1


def test_control_jobs_reports_until_all_jobs_finish(monkeypatch) -> None:
    calls = [
        {
            "submitted": 1,
            "pending": 0,
            "running": 1,
            "finished": 0,
            "active": 1,
            "unknown": 0,
        },
        {
            "submitted": 1,
            "pending": 0,
            "running": 0,
            "finished": 1,
            "active": 0,
            "unknown": 0,
        },
    ]
    reported = []

    monkeypatch.setattr(
        scheduler,
        "get_job_state_counts",
        lambda ids, sched: calls.pop(0),
    )
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)

    scheduler.control_jobs(
        [1],
        "slurm",
        monitor=reported.append,
        poll_interval=0,
    )

    assert [item["active"] for item in reported] == [1, 0]
