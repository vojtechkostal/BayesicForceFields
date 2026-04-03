from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from ..io.utils import extract_train_dir, prepare_path
from .specs import Specs

PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class SimulationSystem:
    """One staged simulation system inside a trainset directory."""

    system_id: str
    fn_topol: Path
    fn_coord: Path


@dataclass(frozen=True, slots=True)
class TrajectorySet:
    """One analyzed sample or reference set spanning one or more trajectories."""

    sample_id: str
    fn_topol: tuple[Path, ...]
    fn_coord: tuple[Path, ...]
    fn_trj: tuple[Path, ...]
    params: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (len(self.fn_topol) == len(self.fn_coord) == len(self.fn_trj)):
            raise ValueError(
                "TrajectorySet requires the same number of topology, coordinate, "
                "and trajectory files."
            )
        if self.params is not None:
            object.__setattr__(
                self,
                "params",
                np.asarray(self.params, dtype=float).reshape(-1),
            )


@dataclass(slots=True)
class TrainSetInfo:
    """Completed simulation samples and staged systems for one trainset."""

    train_dir: Path
    specs: Specs
    systems: list[SimulationSystem]
    samples: list[TrajectorySet]

    @classmethod
    def from_dir(
        cls,
        train_dir: PathLike,
        *,
        strict: bool = True,
    ) -> "TrainSetInfo":
        prepared_dir = prepare_path(Path(train_dir).resolve())
        specs_data, systems_data, samples_data = extract_train_dir(prepared_dir)
        if specs_data is None:
            raise ValueError("Training set is missing specs information.")

        specs = Specs(specs_data)
        systems = [
            SimulationSystem(
                system_id=str(system["system_id"]),
                fn_topol=prepared_dir / system["topology"],
                fn_coord=prepared_dir / system["coordinates"],
            )
            for system in systems_data
        ]

        completed_samples: list[TrajectorySet] = []
        sample_issues: list[str] = []
        for sample_id in sorted((samples_data or {})):
            sample = samples_data[sample_id]
            if not isinstance(sample, dict):
                sample_issues.append(
                    f"sample {sample_id!r}: record must be a mapping, got "
                    f"{type(sample).__name__}."
                )
                continue

            if sample.get("status") != "completed":
                continue
            outputs = sample.get("outputs", [])
            if "params" not in sample:
                sample_issues.append(
                    f"sample {sample_id!r}: completed sample is missing 'params'."
                )
                continue
            if not isinstance(outputs, list):
                sample_issues.append(
                    f"sample {sample_id!r}: 'outputs' must be a list."
                )
                continue
            if any(not isinstance(output, dict) for output in outputs):
                sample_issues.append(
                    f"sample {sample_id!r}: all outputs must be mappings."
                )
                continue
            if any("system_id" not in output for output in outputs):
                sample_issues.append(
                    f"sample {sample_id!r}: each output must define 'system_id'."
                )
                continue
            outputs = sorted(outputs, key=lambda item: item["system_id"])
            if len(outputs) != len(systems):
                sample_issues.append(
                    f"sample {sample_id!r}: expected {len(systems)} outputs, "
                    f"found {len(outputs)}."
                )
                continue
            if any(
                output.get("trajectory") is None
                for output in outputs
            ):
                sample_issues.append(
                    f"sample {sample_id!r}: completed outputs must all define "
                    "trajectory files."
                )
                continue

            fn_trj = tuple(prepared_dir / output["trajectory"] for output in outputs)
            if not all(path.exists() for path in fn_trj):
                missing = [
                    str(path.relative_to(prepared_dir))
                    for path in fn_trj
                    if not path.exists()
                ]
                sample_issues.append(
                    f"sample {sample_id!r}: missing trajectory file(s): "
                    + ", ".join(missing)
                )
                continue

            completed_samples.append(
                TrajectorySet(
                    sample_id=str(sample_id),
                    fn_topol=tuple(system.fn_topol for system in systems),
                    fn_coord=tuple(system.fn_coord for system in systems),
                    fn_trj=fn_trj,
                    params=np.asarray(sample["params"], dtype=float),
                )
            )

        if strict and sample_issues:
            issues = "\n".join(f"- {issue}" for issue in sample_issues)
            raise ValueError(
                "Training set contains invalid completed sample records:\n"
                f"{issues}"
            )
        if strict and not completed_samples:
            raise ValueError(
                f"No completed training samples found in {prepared_dir}. "
                "Check that validation/simulation finished and samples.yaml "
                "contains completed outputs."
            )

        return cls(
            train_dir=prepared_dir,
            specs=specs,
            systems=systems,
            samples=completed_samples,
        )

    @property
    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    @property
    def hashes(self) -> list[str]:
        return self.sample_ids

    @property
    def inputs(self) -> np.ndarray:
        if not self.samples:
            return np.empty((0, 0), dtype=float)
        return np.asarray([sample.params for sample in self.samples], dtype=float)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def fn_topol(self) -> list[Path]:
        return [system.fn_topol for system in self.systems]

    @property
    def fn_coord(self) -> list[Path]:
        return [system.fn_coord for system in self.systems]

    @property
    def fn_trj(self) -> list[list[Path]]:
        if not self.samples:
            return []
        return [list(sample.fn_trj) for sample in self.samples]
