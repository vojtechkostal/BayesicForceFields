"""One campaign sample: apply its parameters and run MD for every system.

Run by the hidden ``bff md <campaign.yaml> <sample_id>`` command, locally or
as one Slurm array task. Results end up in ``samples/<ID>/``. With a scratch
directory, GROMACS runs there and only the results are copied back.

The job can be rerun: systems whose production run is complete are skipped
and an interrupted production run continues from its checkpoint. With ``max_hours``
the production run stops cleanly before that wall time and the sample is
reported as ``incomplete``.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

from ...domain.samples import load_sample_manifest
from ...domain.specs import Specs
from ...gromacs import check_gmx_available, run_md
from ...io.mdp import read_mdp
from ...io.utils import save_yaml
from ...topology import TopologyModifier
from ..config import PathLike, load_config
from .config import SimulationSystemConfig, load_simulation_systems

# grompp warnings accepted for every sample (e.g. a net-charged system).
MAXWARN = 2
# Written into a system's result directory once its production run is complete.
DONE_MARKER = "production.done"
CHECKPOINT_RE = re.compile(r"Writing checkpoint, step (\d+)")


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
    scratch_dir: str | None = None
    max_hours: float | None = None

    @classmethod
    def load(cls, fn_campaign: PathLike, sample_id: str) -> MDJobConfig:
        """Job settings from ``campaign.yaml`` and parameters from ``samples.yaml``."""
        keys = ("gmx_cmd", "store", "cleanup", "scratch_dir", "max_hours", "systems")
        config = load_config(
            fn_campaign, stage="campaign", allowed=keys, required=("gmx_cmd", "systems")
        )
        campaign_dir = config.base_dir
        samples = load_sample_manifest(campaign_dir / "samples.yaml")["samples"]
        if sample_id not in samples:
            raise ValueError(
                f"Sample {sample_id!r} is not in {campaign_dir / 'samples.yaml'}."
            )
        return cls(
            sample_id=sample_id,
            params=[float(value) for value in samples[sample_id]["params"]],
            campaign_dir=campaign_dir,
            fn_specs=campaign_dir / "specs.yaml",
            gmx_cmd=config.string("gmx_cmd"),
            store=config.strings("store", ("xtc",)),
            cleanup=config.boolean("cleanup", False),
            systems=load_simulation_systems(config),
            scratch_dir=config.string("scratch_dir", None),
            max_hours=config.number("max_hours", None, minimum=0),
        )


def write_sample_topology(
    fn_topol: PathLike,
    specs: Specs,
    params: list[float] | np.ndarray,
    fn_out: PathLike,
) -> None:
    """Write a topology with explicit parameters and reconstructed charges."""
    if not specs.is_valid(params).all():
        raise ValueError(
            "Explicit parameter values or reconstructed implicit charges violate "
            "the configured bounds: " + specs.violations(params)
        )
    topology = TopologyModifier(fn_topol)
    topology.apply_parameters(specs.as_dict(specs.complete(params)[0]))
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


def last_checkpoint_step(fn_log: Path) -> int | None:
    """Step of the last checkpoint recorded in an ``mdrun`` log.

    ``mdrun`` writes a checkpoint at its final step, whether the run reached
    ``nsteps`` or was stopped early by ``-maxh``.
    """
    if not fn_log.is_file():
        return None
    steps = CHECKPOINT_RE.findall(fn_log.read_text(errors="ignore"))
    return int(steps[-1]) if steps else None


def production_is_complete(directory: Path, fn_mdp: Path, n_steps: int) -> bool:
    """Whether the production run in ``directory`` reached ``n_steps``.

    A run is complete when its marker exists, when its log records a final
    checkpoint at ``n_steps``, or when its trajectory holds every frame, so
    the check works without a stored trajectory and after ``cleanup``.
    """
    marker = directory / DONE_MARKER
    if marker.is_file() and marker.read_text().strip() == str(n_steps):
        return True
    step = last_checkpoint_step(directory / "production.log")
    if step is not None and step >= n_steps:
        return True
    return trajectory_is_complete(directory / "production.xtc", fn_mdp, n_steps)


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


def main(fn_campaign: PathLike, sample_id: str) -> None:
    started = time.monotonic()
    job = MDJobConfig.load(fn_campaign, sample_id)
    specs = Specs(job.fn_specs)
    check_gmx_available(job.gmx_cmd)
    sample_dir = job.campaign_dir / "samples" / job.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    work_dir = sample_dir
    if job.scratch_dir is not None:
        scratch = os.path.expanduser(os.path.expandvars(job.scratch_dir))
        if "$" in scratch:
            print(
                f"Warning: scratch_dir {job.scratch_dir!r} uses an undefined "
                f"variable; running in {sample_dir} instead.",
                flush=True,
            )
        else:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            work_dir = Path(
                tempfile.mkdtemp(prefix=f"bff-{job.sample_id}-", dir=scratch)
            )
    fn_log = work_dir / "gmx.log"

    def copy_back(
        run_dir: Path, system_dir: Path, suffixes: tuple[str, ...] | None = None
    ) -> None:
        if run_dir == system_dir or not run_dir.is_dir():
            return
        system_dir.mkdir(parents=True, exist_ok=True)
        for path in run_dir.iterdir():
            if path.is_file() and (suffixes is None or path.suffix[1:] in suffixes):
                shutil.copy2(path, system_dir / path.name)

    # Per system: role -> path relative to the campaign directory.
    outputs: dict[str, dict[str, str | list[str]]] = {}
    status = "failed"
    run_dir = system_dir = sample_dir
    try:
        complete = True
        for system in job.systems:
            system_dir = sample_dir / system.system_id
            run_dir = work_dir / system.system_id
            trajectory = system_dir / "production.xtc"
            mdp = system.mdp_production_path
            done = production_is_complete(system_dir, mdp, system.n_steps)
            if not done:
                if run_dir != system_dir and system_dir.is_dir():
                    shutil.copytree(system_dir, run_dir, dirs_exist_ok=True)
                run_dir.mkdir(parents=True, exist_ok=True)
                restart = (run_dir / "production.cpt").is_file()
                max_hours = None
                if job.max_hours is not None:
                    max_hours = job.max_hours - (time.monotonic() - started) / 3600
                    if max_hours < 0.02:
                        complete = False
                        break
                topology = run_dir / "topology.top"
                coordinates = system.coordinates_path
                if not restart:
                    write_sample_topology(
                        system.topology_path, specs, job.params, topology
                    )
                    if system.mdp_em_path is not None:
                        run_md(
                            run_dir / "em",
                            mdp=system.mdp_em_path,
                            topology=topology,
                            coordinates=coordinates,
                            index=system.index_path,
                            gmx_cmd=job.gmx_cmd,
                            maxwarn=MAXWARN,
                            log=fn_log,
                        )
                        coordinates = run_dir / "em.gro"
                run_md(
                    run_dir / "production",
                    mdp=mdp,
                    topology=topology,
                    coordinates=coordinates,
                    index=system.index_path,
                    bias=system.bias,
                    gmx_cmd=job.gmx_cmd,
                    n_steps=system.n_steps,
                    maxwarn=MAXWARN,
                    mdrun_args=("-dlb", "yes"),
                    max_hours=max_hours,
                    restart=restart,
                    log=fn_log,
                )
                done = production_is_complete(run_dir, mdp, system.n_steps)
                # The sample's topology is always kept: QoI routines read it.
                copy_back(
                    run_dir,
                    system_dir,
                    (*job.store, "top") if done and job.cleanup else None,
                )
                if done:
                    (system_dir / DONE_MARKER).write_text(f"{system.n_steps}\n")

            stored: dict[str, str | list[str]] = {
                "topology": str(
                    (system_dir / "topology.top").relative_to(job.campaign_dir)
                )
            }
            if "xtc" in job.store and trajectory.is_file():
                stored["trajectory"] = str(trajectory.relative_to(job.campaign_dir))
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
            outputs[system.system_id] = stored
            if not done:
                complete = False
                break
        status = "completed" if complete else "incomplete"
    finally:
        if status == "failed":
            copy_back(run_dir, system_dir)
        if work_dir != sample_dir:
            if fn_log.is_file():
                with open(sample_dir / "gmx.log", "a", encoding="utf-8") as handle:
                    handle.write(fn_log.read_text(encoding="utf-8"))
            shutil.rmtree(work_dir, ignore_errors=True)
        save_yaml({"status": status, "outputs": outputs}, sample_dir / "result.yaml")
