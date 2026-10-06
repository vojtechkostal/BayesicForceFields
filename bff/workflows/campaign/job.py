"""One campaign sample: apply its parameters and run MD for every system.

Run by the hidden ``bff md <samples/ID/config.yaml>`` command, locally or as
one Slurm array task. All files of the sample stay in ``samples/<ID>/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

from ...domain.specs import ChargeConstraint, Specs
from ...domain.systems import validate_system_id
from ...gromacs import check_gmx_available, run_md
from ...io.mdp import read_mdp
from ...io.utils import load_yaml, save_yaml
from ...topology import TopologyModifier
from ..config import PathLike, check_keys, resolve_path, strict_bool
from .config import SimulationSystemConfig, load_simulation_systems, normalize_store

# grompp warnings accepted for every sample (e.g. a net-charged system).
MAXWARN = 2


@dataclass(frozen=True)
class MDJobConfig:
    sample_id: str
    params: list[float]
    campaign_dir: Path
    fn_specs: Path
    gmx_cmd: str
    store: tuple[str, ...]
    cleanup: bool
    systems: list[SimulationSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> MDJobConfig:
        fn_config = Path(fn_config).resolve()
        keys = ("sample_id", "params", "campaign_dir", "fn_specs", "gmx_cmd", "systems")
        config = check_keys(
            load_yaml(fn_config),
            where="MD job configuration",
            allowed=(*keys, "store", "cleanup"),
            required=keys,
        )
        if not isinstance(config["params"], list):
            raise ValueError("params must be a list of numeric values.")
        base_dir = fn_config.parent
        return cls(
            sample_id=validate_system_id(config["sample_id"], field="sample_id"),
            params=[float(value) for value in config["params"]],
            campaign_dir=resolve_path(base_dir, config["campaign_dir"]),
            fn_specs=resolve_path(base_dir, config["fn_specs"], kind="specs file"),
            gmx_cmd=str(config["gmx_cmd"]),
            store=normalize_store(config.get("store")),
            cleanup=strict_bool(config.get("cleanup", False), field="cleanup"),
            systems=load_simulation_systems(config["systems"], base_dir=base_dir),
        )


def write_sample_topology(
    fn_topol: PathLike,
    specs: Specs,
    params: list[float] | np.ndarray,
    fn_out: PathLike,
) -> None:
    """Write a topology with explicit parameters and reconstructed charges."""
    constraint = ChargeConstraint(specs)
    if not constraint(params).all():
        raise ValueError(
            "Explicit parameter values or reconstructed implicit charges violate "
            "the configured bounds: " + constraint.describe_violations(params)
        )
    values = specs.with_implicit_charges(params).reshape(-1)
    topology = TopologyModifier(fn_topol)
    topology.apply_parameters(specs.parameter_dict(values))
    for charge_constraint in specs.charge_constraints:
        for group in topology.selected_groups(
            charge_constraint.selection, charge_constraint.scope
        ):
            actual = sum(topology.atoms[index].charge for index in group)
            if not np.isclose(actual, charge_constraint.target, atol=1e-8):
                raise ValueError(
                    f"Applied charge constraint {charge_constraint.selection!r} has "
                    f"charge {actual}, expected {charge_constraint.target}."
                )
    topology.write(fn_out)


def trajectory_is_complete(fn_xtc: Path, fn_mdp: Path, n_steps: int) -> bool:
    """Whether the trajectory holds every frame written over ``n_steps``."""
    mdp = read_mdp(fn_mdp)
    stride = int(mdp.get("nstxout-compressed", mdp.get("nstxout_compressed", 0)))
    if stride <= 0 or not fn_xtc.exists():
        return False
    try:
        with XTCReader(str(fn_xtc)) as reader:
            n_frames = reader.n_frames
    except (EOFError, OSError, ValueError):
        return False
    return n_frames >= n_steps // stride + 1


def main(fn_config: PathLike) -> None:
    job = MDJobConfig.load(fn_config)
    specs = Specs(job.fn_specs)
    check_gmx_available(job.gmx_cmd)
    sample_dir = job.campaign_dir / "samples" / job.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    fn_log = sample_dir / "gmx.log"

    outputs: list[dict[str, object]] = []
    complete: list[bool] = []
    status = "failed"
    try:
        for system in job.systems:
            system_dir = sample_dir / system.system_id
            system_dir.mkdir(parents=True, exist_ok=True)
            topology = system_dir / "topology.top"
            write_sample_topology(system.topology_path, specs, job.params, topology)

            coordinates = system.coordinates_path
            if system.mdp_em_path is not None:
                run_md(
                    system_dir / "em",
                    mdp=system.mdp_em_path,
                    topology=topology,
                    coordinates=coordinates,
                    index=system.index_path,
                    gmx_cmd=job.gmx_cmd,
                    maxwarn=MAXWARN,
                    log=fn_log,
                )
                coordinates = system_dir / "em.gro"
            run_md(
                system_dir / "production",
                mdp=system.mdp_production_path,
                topology=topology,
                coordinates=coordinates,
                index=system.index_path,
                bias=system.bias,
                gmx_cmd=job.gmx_cmd,
                n_steps=system.n_steps,
                maxwarn=MAXWARN,
                mdrun_args=("-dlb", "yes"),
                log=fn_log,
            )
            trajectory = system_dir / "production.xtc"
            complete.append(
                trajectory_is_complete(
                    trajectory, system.mdp_production_path, system.n_steps
                )
            )

            stored: dict[str, str | list[str]] = {}
            for suffix in job.store:
                if suffix == "xtc":
                    continue
                paths = [
                    str(path.relative_to(job.campaign_dir))
                    for path in sorted(system_dir.glob(f"*.{suffix}"))
                    if path.is_file()
                ]
                if paths:
                    stored[suffix] = paths[0] if len(paths) == 1 else paths
            outputs.append(
                {
                    "system_id": system.system_id,
                    "trajectory": (
                        str(trajectory.relative_to(job.campaign_dir))
                        if "xtc" in job.store
                        else None
                    ),
                    "inputs": stored,
                }
            )
        if all(complete):
            status = "completed"
    finally:
        if status != "completed" and job.cleanup:
            for system in job.systems:
                for path in (sample_dir / system.system_id).glob("*"):
                    if path.is_file():
                        path.unlink()
        save_yaml(
            {"sample_id": job.sample_id, "status": status, "outputs": outputs},
            sample_dir / "result.yaml",
        )
