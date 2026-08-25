from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...domain.systems import (
    SystemInputs,
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)
from ...io.utils import load_yaml
from ...qoi.routines import (
    AnalysisRoutineConfig,
    normalize_routine_list,
)
from .._shared.config import PathLike, _resolve_path, _strict_bool


@dataclass(frozen=True, slots=True)
class FrameSliceConfig:
    start: int = 1
    stop: int | None = None
    step: int = 1


@dataclass(frozen=True, slots=True)
class QoISystemConfig:
    system_id: str
    inputs: SystemInputs


@dataclass(frozen=True, slots=True)
class QoITrainingSamplesConfig:
    manifest: Path
    system_ids: tuple[str, ...]
    frames: FrameSliceConfig
    workers: int = -1
    progress_stride: int = 10


@dataclass(frozen=True, slots=True)
class QoIReferenceConfig:
    systems: tuple[QoISystemConfig, ...]
    frames: FrameSliceConfig


@dataclass(frozen=True, slots=True)
class QoIOutputConfig:
    directory: Path
    log: Path
    write_raw: bool = False


@dataclass(frozen=True, slots=True)
class BuildQoIDatasetsConfig:
    fn_config: Path
    training_samples: QoITrainingSamplesConfig
    reference: QoIReferenceConfig
    routines: tuple[AnalysisRoutineConfig, ...]
    in_memory: bool
    output: QoIOutputConfig

    @staticmethod
    def _frames(raw: Any, *, field: str, default_start: int = 1) -> FrameSliceConfig:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field} must be a mapping.")
        unknown = set(raw) - {"start", "stop", "step"}
        if unknown:
            raise ValueError(
                f"{field} contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        step = int(raw.get("step", 1))
        if step <= 0:
            raise ValueError(f"{field}.step must be a positive integer.")
        stop = raw.get("stop")
        start = int(raw.get("start", default_start))
        stop_value = None if stop is None else int(stop)
        if start < 0:
            raise ValueError(f"{field}.start must be non-negative.")
        if stop_value is not None and stop_value <= start:
            raise ValueError(f"{field}.stop must be greater than start or null.")
        return FrameSliceConfig(
            start=start,
            stop=stop_value,
            step=step,
        )

    @classmethod
    def load(cls, fn_config: PathLike) -> "BuildQoIDatasetsConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError("Build-qoi-datasets configuration must contain a mapping.")
        unknown_top = set(config) - {
            "training_samples",
            "reference",
            "routines",
            "run",
            "output",
        }
        if unknown_top:
            raise ValueError(
                "Build-qoi-datasets configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown_top))
            )
        required = {"training_samples", "reference", "routines"}
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(
                "Missing required build-qoi-datasets section(s): "
                + ", ".join(repr(key) for key in missing)
            )

        training = config["training_samples"]
        if not isinstance(training, Mapping):
            raise ValueError("training_samples must be a mapping.")
        unknown_training = set(training) - {
            "manifest",
            "systems",
            "frames",
            "workers",
            "progress_stride",
        }
        if unknown_training:
            raise ValueError(
                "training_samples contains unsupported key(s): "
                + ", ".join(sorted(unknown_training))
            )
        if "manifest" not in training or "systems" not in training:
            raise ValueError(
                "training_samples requires 'manifest' and 'systems'."
            )
        selected_raw = training["systems"]
        if not isinstance(selected_raw, list) or not selected_raw:
            raise ValueError("training_samples.systems must be a non-empty list.")
        selected_ids: list[str] = []
        for index, record in enumerate(selected_raw):
            if not isinstance(record, Mapping) or set(record) != {"system_id"}:
                raise ValueError(
                    f"training_samples.systems[{index}] must contain only system_id."
                )
            selected_ids.append(
                validate_system_id(
                    record["system_id"],
                    field=f"training_samples.systems[{index}].system_id",
                )
            )
        validate_unique_system_ids(selected_ids, field="training_samples.systems")

        reference = config["reference"]
        if not isinstance(reference, Mapping):
            raise ValueError("reference must be a mapping.")
        unknown_reference = set(reference) - {"systems", "frames"}
        if unknown_reference:
            raise ValueError(
                "reference contains unsupported key(s): "
                + ", ".join(sorted(unknown_reference))
            )
        reference_raw = reference.get("systems")
        if not isinstance(reference_raw, list) or not reference_raw:
            raise ValueError("reference.systems must be a non-empty list.")
        reference_systems: list[QoISystemConfig] = []
        for index, record in enumerate(reference_raw):
            if not isinstance(record, Mapping):
                raise ValueError(f"reference.systems[{index}] must be a mapping.")
            if set(record) != {"system_id", "inputs"}:
                raise ValueError(
                    f"reference.systems[{index}] requires exactly system_id and inputs."
                )
            system_id = validate_system_id(
                record["system_id"],
                field=f"reference.systems[{index}].system_id",
            )
            inputs_raw = record["inputs"]
            if not isinstance(inputs_raw, Mapping):
                raise ValueError(
                    f"reference.systems[{index}].inputs must be a mapping."
                )
            reference_systems.append(
                QoISystemConfig(
                    system_id=system_id,
                    inputs=resolve_explicit_inputs(
                        inputs_raw,
                        base_dir=base_dir,
                        system_id=system_id,
                        field=f"reference.systems[{index}].inputs",
                    ),
                )
            )
        reference_ids = [system.system_id for system in reference_systems]
        validate_unique_system_ids(reference_ids, field="reference.systems")
        if set(selected_ids) != set(reference_ids):
            missing_ref = sorted(set(selected_ids) - set(reference_ids))
            extra_ref = sorted(set(reference_ids) - set(selected_ids))
            raise ValueError(
                "training_samples.systems and reference.systems must contain "
                "identical system_id sets; "
                f"missing from reference={missing_ref}, extra in reference={extra_ref}."
            )

        routines_raw = config["routines"]
        routines = normalize_routine_list(routines_raw, base_dir=base_dir)
        selected_id_set = set(selected_ids)
        for routine in routines:
            unknown_systems = sorted(set(routine.systems) - selected_id_set)
            if unknown_systems:
                raise ValueError(
                    f"routines.{routine.name}.systems contains unknown system "
                    f"ID(s) {unknown_systems}; expected a subset of "
                    f"{sorted(selected_id_set)}."
                )
        for system in reference_systems:
            applicable = [
                routine for routine in routines if system.system_id in routine.systems
            ]
            if not applicable:
                raise ValueError(
                    f"reference.systems.{system.system_id}: no routine applies to "
                    "this system; add its ID to routines[].systems or remove it."
                )
            roles = set(system.inputs.inputs)
            required_roles = {
                role
                for routine in applicable
                for role in (
                    ("topology", "coordinates", "trajectory")
                    if routine.uses_trajectory
                    else routine.inputs
                )
            }
            missing_roles = sorted(required_roles - roles)
            if missing_roles:
                raise ValueError(
                    f"reference.systems.{system.system_id}.inputs is missing "
                    f"required role(s) {missing_roles}; configured routines need "
                    f"{sorted(required_roles)}."
                )
            supported_roles = {
                "topology",
                "coordinates",
                "trajectory",
                *(role for routine in applicable for role in routine.inputs),
            }
            unsupported_roles = sorted(roles - supported_roles)
            if unsupported_roles:
                raise ValueError(
                    f"reference.systems.{system.system_id}.inputs contains "
                    f"unsupported role(s) {unsupported_roles}; expected only "
                    f"{sorted(supported_roles)}."
                )

        run_raw = config.get("run", {})
        if not isinstance(run_raw, Mapping):
            raise ValueError("run must be a mapping.")
        unknown_run = set(run_raw) - {"in_memory"}
        if unknown_run:
            raise ValueError(
                "run contains unsupported key(s): "
                + ", ".join(sorted(unknown_run))
            )
        output_raw = config.get("output", {})
        if not isinstance(output_raw, Mapping):
            raise ValueError("output must be a mapping.")
        unknown_output = set(output_raw) - {"directory", "log", "write_raw"}
        if unknown_output:
            raise ValueError(
                "output contains unsupported key(s): "
                + ", ".join(sorted(unknown_output))
            )
        output_dir = _resolve_path(
            base_dir,
            output_raw.get("directory", "./qoi"),
            must_exist=False,
            kind="QoI output directory",
        )
        workers = int(training.get("workers", -1))
        if workers == 0 or workers < -1:
            raise ValueError(
                "training_samples.workers must be a positive integer or -1."
            )
        progress_stride = int(training.get("progress_stride", 10))
        if progress_stride <= 0:
            raise ValueError(
                "training_samples.progress_stride must be a positive integer."
            )
        return cls(
            fn_config=fn_config,
            training_samples=QoITrainingSamplesConfig(
                manifest=_resolve_path(
                    base_dir,
                    training["manifest"],
                    kind="sample campaign manifest",
                ),
                system_ids=tuple(selected_ids),
                frames=cls._frames(
                    training.get("frames"), field="training_samples.frames"
                ),
                workers=workers,
                progress_stride=progress_stride,
            ),
            reference=QoIReferenceConfig(
                systems=tuple(reference_systems),
                frames=cls._frames(reference.get("frames"), field="reference.frames"),
            ),
            routines=routines,
            in_memory=_strict_bool(
                run_raw.get("in_memory", True), field="run.in_memory"
            ),
            output=QoIOutputConfig(
                directory=output_dir,
                log=_resolve_path(
                    base_dir,
                    output_raw.get(
                        "log", output_dir.parent / "build-qoi-datasets.log"
                    ),
                    must_exist=False,
                    kind="build-qoi-datasets log file",
                ),
                write_raw=_strict_bool(
                    output_raw.get("write_raw", False), field="output.write_raw"
                ),
            ),
        )
