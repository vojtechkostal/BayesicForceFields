"""The ``samples.yaml`` manifest of a simulation campaign.

Layout (paths relative to the campaign directory)::

    parameter_names: [charge C1, ...]    # order of every ``params`` list
    provenance: {source: ..., seed: ...} # where the parameters came from
    systems:
      <system_id>: {n_steps: 1000}
    samples:
      <sample_id>:
        params: [...]
        status: staged | completed | incomplete | failed
        job_id: 123_4                    # Slurm only
        outputs:
          <system_id>:
            topology: samples/<sample_id>/<system_id>/topology.top
            trajectory: samples/<sample_id>/<system_id>/production.xtc
            pmf: ...                     # every other stored suffix

Inputs shared by all samples are staged under ``systems/<system_id>/`` with
fixed names (``topology.top``, ``coordinates.gro``, ...).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..io.utils import load_yaml, save_yaml
from .systems import PathValue, resolve_explicit_inputs, validate_system_id

STATUSES = ("staged", "completed", "incomplete", "failed")


def write_sample_manifest(
    path: Path,
    *,
    parameter_names: Sequence[str],
    systems: Mapping[str, int],
    samples: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``samples.yaml``; ``systems`` maps system IDs to ``n_steps``."""
    save_yaml(
        {
            "parameter_names": list(parameter_names),
            "provenance": dict(provenance or {}),
            "systems": {
                system_id: {"n_steps": int(n_steps)}
                for system_id, n_steps in systems.items()
            },
            "samples": {
                sample_id: {
                    key: value
                    for key, value in record.items()
                    if value is not None and value != {}
                }
                for sample_id, record in samples.items()
            },
        },
        path,
    )
    return path


def load_sample_manifest(path: Path) -> dict[str, Any]:
    """Read and check ``samples.yaml``."""
    data = load_yaml(path)
    expected = {"parameter_names", "provenance", "systems", "samples"}
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(
            f"{path} is not a campaign manifest: expected the keys "
            f"{sorted(expected)}. Manifests of older BFF versions cannot be read; "
            "rerun the campaign."
        )
    names = data["parameter_names"]
    for system_id in data["systems"]:
        validate_system_id(system_id, field=f"{path}: systems")
    for sample_id, record in data["samples"].items():
        where = f"{path}: samples.{sample_id}"
        if len(record.get("params", ())) != len(names):
            raise ValueError(
                f"{where}.params must have one value per parameter_names entry."
            )
        if record.get("status") not in STATUSES:
            raise ValueError(f"{where}.status must be one of {STATUSES}.")
        unknown = set(record.get("outputs", {})) - set(data["systems"])
        if unknown:
            raise ValueError(
                f"{where}.outputs has unknown system(s) {sorted(unknown)}."
            )
    return dict(data)


@dataclass(frozen=True)
class CampaignSample:
    """One completed sample: its parameters and input files per system."""

    sample_id: str
    params: np.ndarray
    inputs: dict[str, dict[str, PathValue]] = field(repr=False)


@dataclass(frozen=True, repr=False)
class SampleSet:
    """The completed samples of a campaign, ready for analysis.

    ``parameter_names`` name the columns of :attr:`inputs`, ``system_ids`` the
    systems of every sample, and ``unusable`` maps completed samples with
    missing files to the reason.
    """

    campaign_dir: Path
    parameter_names: tuple[str, ...]
    system_ids: tuple[str, ...]
    samples: list[CampaignSample]
    # Completed samples whose files are missing, with the reason.
    unusable: dict[str, str]

    @classmethod
    def from_manifest(cls, fn_manifest: Path) -> SampleSet:
        """Read the completed samples listed in ``fn_manifest``.

        Every system of a sample gets ``coordinates`` and ``topology`` from
        the staged ``systems/<system_id>/``; the sample's own outputs, such as
        its modified ``topology`` and its ``trajectory``, replace or add roles.
        """
        fn_manifest = Path(fn_manifest).resolve()
        campaign_dir = fn_manifest.parent
        manifest = load_sample_manifest(fn_manifest)
        system_ids = tuple(manifest["systems"])
        samples: list[CampaignSample] = []
        unusable: dict[str, str] = {}
        for sample_id, record in manifest["samples"].items():
            if record["status"] != "completed":
                continue
            inputs: dict[str, dict[str, PathValue]] = {}
            try:
                for system_id in system_ids:
                    staged = campaign_dir / "systems" / system_id
                    roles = {
                        "topology": str(staged / "topology.top"),
                        "coordinates": str(staged / "coordinates.gro"),
                    } | record.get("outputs", {}).get(system_id, {})
                    inputs[system_id] = resolve_explicit_inputs(
                        roles,
                        base_dir=campaign_dir,
                        system_id=system_id,
                        field=f"samples.{sample_id}.outputs.{system_id}",
                    ).inputs
            except (ValueError, FileNotFoundError) as exc:
                unusable[str(sample_id)] = str(exc)
                continue
            samples.append(
                CampaignSample(
                    sample_id=str(sample_id),
                    params=np.asarray(record["params"], dtype=float),
                    inputs=inputs,
                )
            )
        return cls(
            campaign_dir=campaign_dir,
            parameter_names=tuple(manifest["parameter_names"]),
            system_ids=system_ids,
            samples=samples,
            unusable=unusable,
        )

    def __repr__(self) -> str:
        return (
            f"SampleSet({len(self.samples)} samples, systems="
            f"{list(self.system_ids)}, {len(self.unusable)} unusable)"
        )

    @property
    def sample_ids(self) -> list[str]:
        """Sample IDs, in manifest order."""
        return [sample.sample_id for sample in self.samples]

    @property
    def inputs(self) -> np.ndarray:
        """Parameters of all samples, shape ``(n_samples, n_parameters)``."""
        return np.asarray(
            [sample.params for sample in self.samples], dtype=float
        ).reshape(len(self.samples), len(self.parameter_names))
