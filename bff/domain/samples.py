"""The ``samples.yaml`` manifest of a simulation campaign and its samples."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from ..io.utils import load_yaml, save_yaml
from .specs import Specs
from .systems import (
    PathValue,
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)

PathLike = Union[str, Path]


def _serialize(value: Any, base_dir: Path) -> Any:
    if isinstance(value, Path):
        path = value.resolve()
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)
    if isinstance(value, (list, tuple)):
        return [_serialize(item, base_dir) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item, base_dir) for key, item in value.items()}
    return value


def load_sample_manifest(filename: str | Path) -> dict[str, Any]:
    """Load and validate sampling campaign handoff metadata."""
    path = Path(filename).resolve()
    data = load_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Sample metadata {path} must contain a mapping.")
    unknown_top = set(data) - {"systems", "samples"}
    if unknown_top:
        raise ValueError(
            f"Sample metadata {path} contains unsupported key(s): "
            + ", ".join(sorted(unknown_top))
        )
    systems = data.get("systems")
    samples = data.get("samples")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"Sample metadata {path}.systems must be a non-empty list.")
    if not isinstance(samples, Mapping):
        raise ValueError(f"Sample metadata {path}.samples must be a mapping.")
    system_ids: list[str] = []
    for index, record in enumerate(systems):
        if not isinstance(record, Mapping) or "system_id" not in record:
            raise ValueError(
                f"Sample metadata {path}.systems[{index}] must define system_id."
            )
        unknown = set(record) - {
            "system_id", "topology", "coordinates", "mdp", "index", "bias", "n_steps"
        }
        if unknown:
            raise ValueError(
                f"Sample metadata {path}.systems[{index}] contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        system_ids.append(
            validate_system_id(record["system_id"], field=f"systems[{index}].system_id")
        )
        for role in ("topology", "coordinates"):
            value = record.get(role)
            resolved = path.parent / value if isinstance(value, str) else None
            if resolved is None or not resolved.resolve().is_file():
                raise ValueError(
                    f"Sample metadata {path}.systems[{index}].{role}: expected an "
                    f"existing relative file path, got {value!r}."
                )
    validate_unique_system_ids(system_ids, field="sample metadata systems")
    return dict(data)


def write_sample_manifest(
    systems: list[Mapping[str, Any]], samples: Mapping[str, Any], filename: str | Path
) -> Path:
    """Write sampling metadata with paths relative to the metadata file."""
    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    system_ids = [
        validate_system_id(record.get("system_id"), field="systems[].system_id")
        for record in systems
    ]
    validate_unique_system_ids(system_ids, field="sample metadata systems")
    save_yaml(
        {
            "systems": _serialize(systems, path.parent),
            "samples": _serialize(samples, path.parent),
        },
        path,
    )
    return path


@dataclass(frozen=True, slots=True)
class SimulationSystem:
    """One staged simulation system inside a sampling campaign."""

    system_id: str
    topology_path: Path
    coordinates_path: Path


@dataclass(frozen=True, slots=True)
class TrajectorySet:
    """One analyzed sample or reference set spanning one or more trajectories."""

    sample_id: str
    system_ids: tuple[str, ...]
    topology_paths: tuple[Path, ...]
    coordinate_paths: tuple[Path, ...]
    trajectory_paths: tuple[Path, ...]
    params: np.ndarray | None = None
    input_roles: tuple[dict[str, PathValue], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (
            len(self.system_ids)
            == len(self.topology_paths)
            == len(self.coordinate_paths)
            == len(self.trajectory_paths)
        ):
            raise ValueError(
                'TrajectorySet requires the same number of system IDs, topology, '
                'coordinate, and trajectory files.'
            )
        validate_unique_system_ids(list(self.system_ids), field='system_ids')
        if self.input_roles and len(self.input_roles) != len(self.system_ids):
            raise ValueError(
                'TrajectorySet input role mappings must match system_ids.'
            )
        if not self.input_roles:
            object.__setattr__(
                self,
                'input_roles',
                tuple(
                    {
                        'topology': topology,
                        'coordinates': coordinates,
                        'trajectory': trajectory,
                    }
                    for topology, coordinates, trajectory in zip(
                        self.topology_paths,
                        self.coordinate_paths,
                        self.trajectory_paths,
                    )
                ),
            )
        if self.params is not None:
            object.__setattr__(
                self,
                'params',
                np.asarray(self.params, dtype=float).reshape(-1),
            )


@dataclass(slots=True)
class SampleSet:
    """Completed simulation samples and staged systems for one sampling campaign."""

    campaign_dir: Path
    specs: Specs
    systems: list[SimulationSystem]
    samples: list[TrajectorySet]

    @classmethod
    def from_dir(
        cls,
        campaign_dir: PathLike,
        *,
        manifest: PathLike | None = None,
        strict: bool = True,
    ) -> 'SampleSet':
        prepared_dir = Path(campaign_dir).resolve()
        if not prepared_dir.is_dir():
            raise ValueError(
                f"Sampling campaign must be an existing directory: {prepared_dir}."
            )
        specs_path = prepared_dir / 'specs.yaml'
        if not specs_path.is_file():
            raise ValueError(
                f"Sampling campaign is missing specifications: {specs_path}."
            )
        manifest_path = (
            prepared_dir / 'samples.yaml'
            if manifest is None
            else Path(manifest).resolve()
        )
        if manifest_path.parent != prepared_dir:
            raise ValueError(
                f"Sample manifest {manifest_path} must be located in campaign "
                f"directory {prepared_dir} beside specs.yaml."
            )
        campaign = load_sample_manifest(manifest_path)
        systems_data = campaign['systems']
        samples_data = campaign['samples']
        specs = Specs(specs_path)
        systems = [
            SimulationSystem(
                system_id=validate_system_id(
                    system['system_id'], field='samples.systems[].system_id'
                ),
                topology_path=prepared_dir / system['topology'],
                coordinates_path=prepared_dir / system['coordinates'],
            )
            for system in systems_data
        ]
        expected_system_ids = [system.system_id for system in systems]
        validate_unique_system_ids(
            expected_system_ids, field='samples.systems'
        )
        systems_by_id = {system.system_id: system for system in systems}

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

            if sample.get('status') != 'completed':
                continue
            outputs = sample.get('outputs', [])
            if 'params' not in sample:
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
            unsupported_output_keys = sorted(
                {
                    key
                    for output in outputs
                    for key in set(output) - {'system_id', 'trajectory', 'inputs'}
                }
            )
            if unsupported_output_keys:
                sample_issues.append(
                    f"sample {sample_id!r}: outputs contain unsupported key(s) "
                    f"{unsupported_output_keys}; put additional named file roles "
                    "under outputs[].inputs."
                )
                continue
            if any('system_id' not in output for output in outputs):
                sample_issues.append(
                    f"sample {sample_id!r}: each output must define 'system_id'."
                )
                continue
            output_ids = [
                validate_system_id(
                    output['system_id'],
                    field=f"samples.{sample_id}.outputs[].system_id",
                )
                for output in outputs
            ]
            duplicate_ids = sorted(
                {value for value in output_ids if output_ids.count(value) > 1}
            )
            missing_ids = sorted(set(expected_system_ids) - set(output_ids))
            extra_ids = sorted(set(output_ids) - set(expected_system_ids))
            if duplicate_ids or missing_ids or extra_ids:
                details = []
                if missing_ids:
                    details.append('missing ' + ', '.join(missing_ids))
                if extra_ids:
                    details.append('extra ' + ', '.join(extra_ids))
                if duplicate_ids:
                    details.append('duplicate ' + ', '.join(duplicate_ids))
                sample_issues.append(
                    f"sample {sample_id!r}: system outputs do not match campaign "
                    f"systems ({'; '.join(details)})."
                )
                continue
            outputs_by_id = {
                output['system_id']: output for output in outputs
            }
            outputs = [outputs_by_id[system_id] for system_id in expected_system_ids]
            if any(output.get('trajectory') is None for output in outputs):
                sample_issues.append(
                    f"sample {sample_id!r}: completed outputs must all define "
                    'trajectory files.'
                )
                continue

            fn_trj = tuple(prepared_dir / output['trajectory'] for output in outputs)
            if not all(path.exists() for path in fn_trj):
                missing = [
                    str(path.relative_to(prepared_dir))
                    for path in fn_trj
                    if not path.exists()
                ]
                sample_issues.append(
                    f"sample {sample_id!r}: missing trajectory file(s): "
                    + ', '.join(missing)
                )
                continue

            role_mappings: list[dict[str, PathValue]] = []
            for system_id, output in zip(expected_system_ids, outputs):
                system = systems_by_id[system_id]
                extra_inputs = output.get('inputs', {})
                if not isinstance(extra_inputs, dict):
                    sample_issues.append(
                        f"sample {sample_id!r}, system {system_id!r}: inputs must "
                        "be a mapping."
                    )
                    break
                direct_roles = {
                    key: value
                    for key, value in output.items()
                    if key not in {'system_id', 'inputs'}
                }
                resolved = resolve_explicit_inputs(
                    direct_roles | extra_inputs,
                    base_dir=prepared_dir,
                    system_id=system_id,
                    field=f"samples.{sample_id}.outputs.{system_id}",
                )
                role_mappings.append(
                    {
                        'topology': system.topology_path,
                        'coordinates': system.coordinates_path,
                        **resolved.inputs,
                    }
                )
            if len(role_mappings) != len(expected_system_ids):
                continue

            completed_samples.append(
                TrajectorySet(
                    sample_id=str(sample_id),
                    system_ids=tuple(expected_system_ids),
                    topology_paths=tuple(
                        systems_by_id[system_id].topology_path
                        for system_id in expected_system_ids
                    ),
                    coordinate_paths=tuple(
                        systems_by_id[system_id].coordinates_path
                        for system_id in expected_system_ids
                    ),
                    trajectory_paths=fn_trj,
                    params=np.asarray(sample['params'], dtype=float),
                    input_roles=tuple(role_mappings),
                )
            )

        if strict and sample_issues:
            issues = '\n'.join(f'- {issue}' for issue in sample_issues)
            raise ValueError(
                'Sampling campaign contains invalid completed sample records:\n'
                f'{issues}'
            )
        if strict and not completed_samples:
            raise ValueError(
                f'No completed sampling results found in {prepared_dir}. '
                'Check that the campaign finished and samples.yaml contains '
                'completed outputs.'
            )

        return cls(
            campaign_dir=prepared_dir,
            specs=specs,
            systems=systems,
            samples=completed_samples,
        )

    @property
    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    @property
    def inputs(self) -> np.ndarray:
        if not self.samples:
            return np.empty((0, 0), dtype=float)
        return np.asarray([sample.params for sample in self.samples], dtype=float)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def trajectory_paths(self) -> list[list[Path]]:
        if not self.samples:
            return []
        return [list(sample.trajectory_paths) for sample in self.samples]
