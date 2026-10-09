from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ...domain.systems import (
    SystemInputs,
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)
from ...qoi.routines import (
    BUILTIN_ROUTINES,
    AnalysisRoutineConfig,
    load_custom_routine,
)
from ..config import ConfigSection, PathLike, load_config


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


@dataclass(frozen=True, slots=True)
class QoIReferenceConfig:
    systems: tuple[QoISystemConfig, ...]
    frames: FrameSliceConfig


@dataclass(frozen=True, slots=True)
class QoIOutputConfig:
    directory: Path
    log: Path


@dataclass(frozen=True, slots=True)
class BuildQoIDatasetsConfig:
    fn_config: Path
    training_samples: QoITrainingSamplesConfig
    reference: QoIReferenceConfig
    routines: tuple[AnalysisRoutineConfig, ...]
    in_memory: bool
    output: QoIOutputConfig

    @classmethod
    def load(cls, fn_config: PathLike) -> BuildQoIDatasetsConfig:
        config = load_config(
            fn_config,
            stage="build-qoi-datasets",
            allowed=("training_samples", "reference", "routines", "run", "output"),
            required=("training_samples", "reference", "routines"),
        )
        training = config.section(
            "training_samples",
            allowed=("manifest", "systems", "frames", "workers"),
            required=("manifest", "systems"),
        )
        selected_ids = [
            validate_system_id(record.get("system_id"), field=record.field("system_id"))
            for record in training.sections(
                "systems", allowed=("system_id",), required=("system_id",)
            )
        ]
        validate_unique_system_ids(selected_ids, field="training_samples.systems")

        reference = config.section(
            "reference", allowed=("systems", "frames"), required=("systems",)
        )
        reference_systems: list[QoISystemConfig] = []
        for record in reference.sections(
            "systems",
            allowed=("system_id", "inputs"),
            required=("system_id", "inputs"),
        ):
            system_id = validate_system_id(
                record.get("system_id"), field=record.field("system_id")
            )
            reference_systems.append(
                QoISystemConfig(
                    system_id=system_id,
                    inputs=resolve_explicit_inputs(
                        record.mapping("inputs"),
                        base_dir=config.base_dir,
                        system_id=system_id,
                        field=record.field("inputs"),
                    ),
                )
            )
        reference_ids = [system.system_id for system in reference_systems]
        validate_unique_system_ids(reference_ids, field="reference.systems")
        if set(selected_ids) != set(reference_ids):
            raise ValueError(
                "training_samples.systems and reference.systems must list the same "
                "system IDs; missing from reference: "
                f"{sorted(set(selected_ids) - set(reference_ids))}, only in "
                f"reference: {sorted(set(reference_ids) - set(selected_ids))}."
            )

        routines = load_routines(config)
        for routine in routines:
            unknown_systems = sorted(set(routine.systems) - set(selected_ids))
            if unknown_systems:
                raise ValueError(
                    f"routines.{routine.name}.systems contains unknown system "
                    f"ID(s) {unknown_systems}; expected a subset of "
                    f"{sorted(selected_ids)}."
                )
        for system in reference_systems:
            _check_reference_roles(system, routines)

        run = config.section("run", allowed=("in_memory",))
        output = config.section("output", allowed=("directory", "log"))
        output_dir = output.path("directory", "./qoi", must_exist=False)
        return cls(
            fn_config=Path(fn_config).resolve(),
            training_samples=QoITrainingSamplesConfig(
                manifest=training.path("manifest"),
                system_ids=tuple(selected_ids),
                frames=_load_frames(training),
                workers=training.integer("workers", -1, minimum=1, special=(-1,)),
            ),
            reference=QoIReferenceConfig(
                systems=tuple(reference_systems),
                frames=_load_frames(reference),
            ),
            routines=routines,
            in_memory=run.boolean("in_memory", True),
            output=QoIOutputConfig(
                directory=output_dir,
                log=output.path(
                    "log",
                    output_dir.parent / "build-qoi-datasets.log",
                    must_exist=False,
                ),
            ),
        )


def _load_frames(parent: ConfigSection) -> FrameSliceConfig:
    frames = parent.section("frames", allowed=("start", "stop", "step"))
    start = frames.integer("start", 1, minimum=0)
    stop = frames.integer("stop", None, minimum=start + 1)
    return FrameSliceConfig(
        start=start, stop=stop, step=frames.integer("step", 1, minimum=1)
    )


def required_input_roles(routines: Iterable[AnalysisRoutineConfig]) -> set[str]:
    """Input roles that the given routines read for one system."""
    return {
        role
        for routine in routines
        for role in (
            ("topology", "coordinates", "trajectory")
            if routine.uses_trajectory
            else routine.inputs
        )
    }


def _check_reference_roles(
    system: QoISystemConfig, routines: tuple[AnalysisRoutineConfig, ...]
) -> None:
    """Require exactly the input roles that this system's routines use."""
    applicable = [
        routine for routine in routines if system.system_id in routine.systems
    ]
    field = f"reference.systems.{system.system_id}"
    if not applicable:
        raise ValueError(
            f"{field}: no routine applies to this system; add its ID to "
            "routines[].systems or remove it."
        )
    roles = set(system.inputs.inputs)
    required_roles = required_input_roles(applicable)
    missing_roles = sorted(required_roles - roles)
    if missing_roles:
        raise ValueError(
            f"{field}.inputs is missing role(s) {missing_roles}; its routines "
            f"need {sorted(required_roles)}."
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
            f"{field}.inputs contains unsupported role(s) {unsupported_roles}; "
            f"expected only {sorted(supported_roles)}."
        )


def load_routines(config: ConfigSection) -> tuple[AnalysisRoutineConfig, ...]:
    """Read ``routines``; built-in ones take ``selections``, custom ones ``inputs``."""
    routines: list[AnalysisRoutineConfig] = []
    for raw in config.sections(
        "routines",
        allowed=(
            "name", "type", "callable", "systems", "selections", "inputs", "options"
        ),
        required=("name", "systems"),
    ):
        name = validate_system_id(raw.get("name"), field=raw.field("name"))
        if ("type" in raw) == ("callable" in raw):
            raise ValueError(
                f"{raw.where} must define exactly one of type or callable."
            )
        systems = raw.strings("systems")
        for system_id in systems:
            validate_system_id(system_id, field=raw.field("systems"))
        validate_unique_system_ids(list(systems), field=raw.field("systems"))
        selections = raw.mapping("selections", {})
        if not all(isinstance(value, str) for value in selections.values()):
            raise ValueError(
                f"{raw.field('selections')} must map names to selection strings."
            )
        options = raw.mapping("options", {})
        duplicated = sorted(set(selections) & set(options))
        if duplicated:
            raise ValueError(
                f"{raw.where} defines {duplicated} in both selections and options."
            )
        inputs = raw.strings("inputs", ())
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"{raw.field('inputs')} contains duplicates.")

        callable_spec = None
        if "type" in raw:
            raw.string("type", choices=BUILTIN_ROUTINES)
            if inputs:
                raise ValueError(
                    f"{raw.field('inputs')} is only supported by custom routines."
                )
        else:
            if selections:
                raise ValueError(
                    f"{raw.field('selections')} is only supported by built-in "
                    "routines; pass custom settings through options."
                )
            callable_spec, _ = load_custom_routine(
                raw.string("callable"), config.base_dir
            )
        routines.append(
            AnalysisRoutineConfig(
                name=name,
                systems=systems,
                type=raw.get("type"),
                callable=callable_spec,
                inputs=inputs,
                options={**selections, **options},
            )
        )
    names = [routine.name for routine in routines]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("routines names must be unique: " + ", ".join(duplicates))
    return tuple(routines)
