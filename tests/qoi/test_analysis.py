import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from bff.qoi import analysis
from bff.qoi.data import QoI
from bff.qoi.routines import AnalysisRoutineConfig


class _Trajectory:
    def __init__(self, frame_count: int, unreadable: set[int]) -> None:
        self.frame_count = frame_count
        self.unreadable = unreadable

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, frame: int) -> SimpleNamespace:
        if frame in self.unreadable:
            raise OSError("incomplete trajectory record")
        return SimpleNamespace(frame=frame)

    def close(self) -> None:
        self.closed = True


class _Universe:
    def __init__(self, frame_count: int, unreadable: set[int]) -> None:
        self.dimensions = None
        self.trajectory = _Trajectory(frame_count, unreadable)
        self.transfer_arguments: tuple[int, int | None, int] | None = None

    def load_new(self, trajectory: str) -> None:
        self.loaded_trajectory = trajectory

    def transfer_to_memory(
        self, *, start: int, stop: int | None, step: int
    ) -> None:
        self.transfer_arguments = (start, stop, step)


@pytest.mark.parametrize(
    ("frame_count", "unreadable", "expected_stop"),
    [
        (12, set(), 12),
        (8, {7}, 7),
        (15, {13, 14}, 13),
    ],
)
def test_analysis_uses_all_readable_frames_for_each_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    unreadable: set[int],
    expected_stop: int,
) -> None:
    universe = _Universe(frame_count, unreadable)
    monkeypatch.setattr(analysis, "prepare_universe", lambda *args, **kwargs: universe)
    monkeypatch.setattr(
        analysis,
        "run_analysis_routine",
        lambda *args, **kwargs: QoI(name="rdf", values=[1.0]),
    )
    routine = AnalysisRoutineConfig(
        name="rdf",
        systems=("acetate",),
        type="rdf",
    )
    task = (
        "reference",
        {
            "acetate": {
                "topology": Path("topology.top"),
                "coordinates": Path("coordinates.gro"),
                "trajectory": Path("trajectory.xtc"),
            }
        },
    )

    if unreadable:
        with pytest.warns(RuntimeWarning, match="all readable frames"):
            analysis.analyze_sample(
                task,
                routines_by_system={"acetate": (routine,)},
                start=0,
                stop=None,
                step=1,
                in_memory=True,
            )
    else:
        analysis.analyze_sample(
            task,
            routines_by_system={"acetate": (routine,)},
            start=0,
            stop=None,
            step=1,
            in_memory=True,
        )

    assert universe.transfer_arguments == (0, expected_stop, 1)


def test_streamed_analysis_uses_the_readable_trajectory_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _Universe(10, {9})
    received: dict[str, int | None] = {}
    monkeypatch.setattr(analysis, "prepare_universe", lambda *args, **kwargs: universe)

    def run_routine(*args, **kwargs):
        received["stop"] = kwargs["stop"]
        return QoI(name="rdf", values=[1.0])

    monkeypatch.setattr(analysis, "run_analysis_routine", run_routine)
    routine = AnalysisRoutineConfig(
        name="rdf",
        systems=("acetate",),
        type="rdf",
    )

    with pytest.warns(RuntimeWarning, match="all readable frames"):
        analysis.analyze_sample(
            (
                "reference",
                {
                    "acetate": {
                        "topology": Path("topology.top"),
                        "coordinates": Path("coordinates.gro"),
                        "trajectory": Path("trajectory.xtc"),
                    }
                },
            ),
            routines_by_system={"acetate": (routine,)},
            start=2,
            stop=None,
            step=1,
            in_memory=False,
        )

    assert received["stop"] == 9


def test_analysis_does_not_warn_when_last_frame_is_outside_stride(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _Universe(14, set())
    monkeypatch.setattr(analysis, "prepare_universe", lambda *args, **kwargs: universe)
    monkeypatch.setattr(
        analysis,
        "run_analysis_routine",
        lambda *args, **kwargs: QoI(name="rdf", values=[1.0]),
    )
    routine = AnalysisRoutineConfig(
        name="rdf",
        systems=("acetate",),
        type="rdf",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        analysis.analyze_sample(
            (
                "reference",
                {
                    "acetate": {
                        "topology": Path("topology.top"),
                        "coordinates": Path("coordinates.gro"),
                        "trajectory": Path("trajectory.xtc"),
                    }
                },
            ),
            routines_by_system={"acetate": (routine,)},
            start=2,
            stop=None,
            step=5,
            in_memory=True,
        )

    assert universe.transfer_arguments == (2, 13, 5)


def test_sample_reuses_one_universe_for_all_trajectory_routines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _Universe(4, set())
    opened = 0

    def prepare(*args, **kwargs):
        nonlocal opened
        opened += 1
        return universe

    received_universes = []

    def run_routine(*args, **kwargs):
        received_universes.append(kwargs["universe"])
        return QoI(name="result", values=[1.0])

    monkeypatch.setattr(analysis, "prepare_universe", prepare)
    monkeypatch.setattr(analysis, "run_analysis_routine", run_routine)
    routines = (
        AnalysisRoutineConfig(name="rdf", systems=("acetate",), type="rdf"),
        AnalysisRoutineConfig(
            name="hb", systems=("acetate",), type="hydrogen_bonds"
        ),
    )

    sample_id, results = analysis.analyze_sample(
        (
            "sample-0",
            {
                "acetate": {
                    "topology": Path("topology.top"),
                    "coordinates": Path("coordinates.gro"),
                    "trajectory": Path("trajectory.xtc"),
                }
            },
        ),
        routines_by_system={"acetate": routines},
        start=0,
        stop=None,
        step=1,
        in_memory=True,
    )

    assert sample_id == "sample-0"
    assert tuple(results["acetate"]) == ("rdf", "hb")
    assert opened == 1
    assert received_universes == [universe, universe]
    assert universe.trajectory.closed


def test_file_only_system_does_not_open_a_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis,
        "prepare_universe",
        lambda *args, **kwargs: pytest.fail("file routine opened a Universe"),
    )
    monkeypatch.setattr(
        analysis,
        "run_analysis_routine",
        lambda *args, **kwargs: QoI(name="pmf", values=[1.0]),
    )
    routine = AnalysisRoutineConfig(
        name="pmf",
        systems=("acetate-calcium",),
        callable="pmf:load",
        inputs=("pmf",),
    )

    sample_id, results = analysis.analyze_sample(
        (
            "reference",
            {"acetate-calcium": {"pmf": Path("profile.pmf")}},
        ),
        routines_by_system={"acetate-calcium": (routine,)},
        start=0,
        stop=None,
        step=1,
        in_memory=True,
    )

    assert sample_id == "reference"
    assert tuple(results["acetate-calcium"]) == ("pmf",)


def test_parallel_analysis_respects_available_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_sizes: list[int] = []

    class Executor:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            assert mp_context == "spawn-context"
            size = max_workers
            pool_sizes.append(size)

        def map(self, function, tasks, chunksize):
            assert chunksize == 1
            return map(function, tasks)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    class Logger:
        def status(self, *args, **kwargs) -> None:
            pass

        def progress_status(self, *args, **kwargs) -> None:
            pass

        def result_summary(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(analysis.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(
        analysis.mp,
        "get_context",
        lambda method: "spawn-context" if method == "spawn" else pytest.fail(method),
    )
    monkeypatch.setattr(analysis, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        analysis,
        "run_analysis_routine",
        lambda *args, **kwargs: QoI(name="pmf", values=[1.0]),
    )
    routine = AnalysisRoutineConfig(
        name="pmf",
        systems=("acetate",),
        callable="pmf:load",
        inputs=("pmf",),
    )

    results = analysis.analyze_samples(
        [
            ("sample-0", {"acetate": {"pmf": Path("zero.pmf")}}),
            ("sample-1", {"acetate": {"pmf": Path("one.pmf")}}),
            ("sample-2", {"acetate": {"pmf": Path("two.pmf")}}),
        ],
        routines_by_system={"acetate": (routine,)},
        start=0,
        stop=None,
        step=1,
        workers=-1,
        progress_stride=1,
        progress_label="Training QoI",
        logger=Logger(),
        in_memory=True,
    )

    assert pool_sizes == [2]
    assert list(results) == ["sample-0", "sample-1", "sample-2"]
