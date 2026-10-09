"""Stage, run, and collect a simulation campaign over parameter samples.

Layout of a campaign directory::

    specs.yaml                       parameter specification
    samples.yaml                     parameters, status, and outputs per sample
    campaign.yaml                    job settings shared by every sample
    systems/<system_id>/             staged inputs shared by all samples
    samples/<sample_id>/             run.out and gmx.log of the sample's job
    samples/<sample_id>/<system_id>/ the sample's topology and MD outputs
    run.sh, tasks.txt, slurm/        Slurm job script, task list, and output

Each sample runs as ``bff md campaign.yaml <sample_id>``.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from gmxtopology import Topology

from ... import slurm
from ...domain.samples import load_sample_manifest, write_sample_manifest
from ...domain.specs import Specs
from ...io.logs import Logger
from ...io.utils import compress_results, load_yaml, save_yaml
from .config import SimulationCampaignConfig, SimulationSystemConfig
from .job import DONE_MARKER, write_sample_topology

# Campaign files and directories; overwrite removes exactly these.
OWNED_PATHS = (
    "specs.yaml",
    "samples.yaml",
    "campaign.yaml",
    "run.sh",
    "tasks.txt",
    "systems",
    "samples",
    "slurm",
)

ParameterDraw = Callable[[], tuple[np.ndarray, dict[str, Any]]]


def fresh_seed() -> int:
    """A new random seed, recorded so that the draw can be repeated."""
    return int(np.random.SeedSequence().generate_state(1)[0])


def stage_systems(
    systems: list[SimulationSystemConfig],
    campaign_dir: Path,
) -> list[SimulationSystemConfig]:
    """Copy every system's inputs to ``systems/<system_id>/`` of the campaign."""
    staged: list[SimulationSystemConfig] = []
    for system in systems:
        system_dir = campaign_dir / "systems" / system.system_id
        system_dir.mkdir(parents=True, exist_ok=True)
        topology = system_dir / "topology.top"
        Topology(system.topology_path).write(topology, overwrite=True)
        mdp_em = None
        if system.mdp_em_path is not None:
            mdp_em = Path(shutil.copy2(system.mdp_em_path, system_dir / "em.mdp"))
        bias = system.bias
        if bias.input_file is not None:
            bias = type(bias).load(
                shutil.copy2(bias.input_file, system_dir / bias.input_filename)
            )
        staged.append(
            SimulationSystemConfig(
                system_id=system.system_id,
                topology_path=topology,
                coordinates_path=Path(
                    shutil.copy2(
                        system.coordinates_path, system_dir / "coordinates.gro"
                    )
                ),
                mdp_em_path=mdp_em,
                mdp_production_path=Path(
                    shutil.copy2(
                        system.mdp_production_path, system_dir / "production.mdp"
                    )
                ),
                index_path=Path(
                    shutil.copy2(system.index_path, system_dir / "index.ndx")
                ),
                bias=bias,
                n_steps=system.n_steps,
            )
        )
    return staged


def prepare_campaign_dir(config: SimulationCampaignConfig) -> None:
    """Refuse to mix campaigns: an existing one must be resumed or overwritten."""
    campaign_dir = config.campaign_dir
    if config.resume:
        if not (campaign_dir / "samples.yaml").is_file():
            raise FileNotFoundError(
                f"resume: {campaign_dir} contains no campaign to resume "
                "(samples.yaml is missing)."
            )
        return
    existing = [name for name in OWNED_PATHS if (campaign_dir / name).exists()]
    if not existing:
        return
    if not config.overwrite:
        raise FileExistsError(
            f"{campaign_dir} already contains a campaign ({', '.join(existing)}). "
            "Set resume: true to continue it or overwrite: true to replace it."
        )
    for name in existing:
        path = campaign_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def check_parameter_samples(specs: Specs, parameter_samples: np.ndarray) -> None:
    """Reject samples that violate the bounds once implicit charges are set."""
    names = specs.explicit_names
    if parameter_samples.ndim != 2 or parameter_samples.shape[1] != len(names):
        raise ValueError(
            f"Parameter samples must have one column per explicit parameter "
            f"({len(names)}), got shape {parameter_samples.shape}."
        )
    if len(parameter_samples) == 0:
        raise ValueError("No parameter samples to simulate.")
    valid = specs.is_valid(parameter_samples)
    if not valid.all():
        invalid = np.flatnonzero(~valid)
        raise ValueError(
            f"{len(invalid)} of {len(parameter_samples)} parameter samples "
            f"(indices {invalid[:10].tolist()}) violate the parameter bounds:\n"
            + specs.violations(parameter_samples[~valid])
        )


def _read_result(campaign_dir: Path, sample_id: str) -> dict[str, Any]:
    """The ``result.yaml`` a sample's job left, if it has not been collected."""
    fn_result = campaign_dir / "samples" / sample_id / "result.yaml"
    return load_yaml(fn_result) if fn_result.is_file() else {}


def _forget_results(campaign_dir: Path, sample_ids: list[str]) -> None:
    """Delete earlier results, so a job that dies without one reads as failed."""
    for sample_id in sample_ids:
        (campaign_dir / "samples" / sample_id / "result.yaml").unlink(missing_ok=True)


def _resumed_samples(
    config: SimulationCampaignConfig, specs: Specs
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Samples and provenance of the staged campaign that is resumed."""
    campaign_dir = config.campaign_dir
    if Specs(campaign_dir / "specs.yaml").to_dict() != specs.to_dict():
        raise ValueError(
            "resume: the parameter specification differs from the staged "
            f"{campaign_dir / 'specs.yaml'}; set overwrite: true to start a new "
            "campaign."
        )
    manifest = load_sample_manifest(campaign_dir / "samples.yaml")
    staged = {
        system_id: record["n_steps"]
        for system_id, record in manifest["systems"].items()
    }
    configured = {system.system_id: system.n_steps for system in config.systems}
    if staged != configured:
        raise ValueError(
            f"resume: systems and n_steps {configured} differ from the staged "
            f"campaign {staged}; set overwrite: true to start a new campaign."
        )
    samples = {
        sample_id: dict(record) | _read_result(campaign_dir, sample_id)
        for sample_id, record in manifest["samples"].items()
    }
    if not samples:
        raise ValueError(f"resume: {campaign_dir / 'samples.yaml'} has no samples.")
    return samples, manifest["provenance"]


def _write_manifest(
    config: SimulationCampaignConfig,
    specs: Specs,
    samples: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    write_sample_manifest(
        config.campaign_dir / "samples.yaml",
        parameter_names=specs.explicit_names,
        systems={system.system_id: system.n_steps for system in config.systems},
        samples={
            sample_id: {
                "params": sample["params"],
                "status": sample["status"],
                "job_id": sample.get("job_id"),
                "outputs": sample.get("outputs"),
            }
            for sample_id, sample in samples.items()
        },
        provenance=provenance,
    )


def _cleanup(
    campaign_dir: Path,
    samples: dict[str, dict[str, Any]],
    systems: list[SimulationSystemConfig],
    store: tuple[str, ...],
) -> None:
    """Keep only stored suffixes and each sample's topology.

    A system without a completed production run is left untouched: its
    checkpoint, log, and ``.tpr`` are what a later ``resume: true`` needs to
    continue it, and pruning them here would force it to restart from
    scratch.
    """
    keep = {f".{suffix}" for suffix in store} | {".top", Path(DONE_MARKER).suffix}
    for sample_id in samples:
        for system in systems:
            system_dir = campaign_dir / "samples" / sample_id / system.system_id
            if not (system_dir / DONE_MARKER).is_file():
                continue
            for path in system_dir.glob("*"):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.suffix not in keep:
                    path.unlink()


def _run_local(
    fn_campaign: Path,
    sample_ids: list[str],
    *,
    max_parallel_jobs: int,
    logger: Logger,
) -> None:
    """Run samples as ``bff md`` processes, at most ``max_parallel_jobs`` at once."""
    campaign_dir = fn_campaign.parent

    def run(sample_id: str) -> int:
        sample_dir = campaign_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Appended, like on Slurm, so a resumed sample keeps its earlier output.
        with open(sample_dir / "run.out", "a", encoding="utf-8") as output:
            return subprocess.run(
                [sys.executable, "-m", "bff.cli", "md", str(fn_campaign), sample_id],
                cwd=campaign_dir,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode

    n_total = len(sample_ids)
    width = len(str(n_total))
    logger.progress_status(
        f"Running MD: {0:>{width}d}/{n_total}",
        0,
        n_total,
        overwrite=True,
        write_file=False,
    )
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as pool:
        futures = {pool.submit(run, sample_id): sample_id for sample_id in sample_ids}
        for done, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            returncode = future.result()
            if returncode != 0:
                logger.warn(
                    f"Sample {sample_id} failed (exit code {returncode}); see "
                    f"samples/{sample_id}/run.out."
                )
            logger.progress_status(
                f"Running MD: {done:>{width}d}/{n_total}",
                done,
                n_total,
                overwrite=True,
                write_file=False,
            )


def _log_summary(
    config: SimulationCampaignConfig,
    logger: Logger,
    *,
    specs: Specs,
    n_samples: int,
    n_pending: int,
    provenance: dict[str, Any],
) -> None:
    logger.kv("Config", config.fn_config)
    logger.kv("Campaign", config.campaign_dir)
    logger.kv("Systems", ", ".join(system.system_id for system in config.systems))
    logger.kv(
        "Samples",
        f"{n_samples} ({n_pending} to run)" if config.resume else n_samples,
    )
    # Full paths and hashes are in samples.yaml; show the settings briefly.
    details = [
        f"{key} {Path(value).name if key in ('parameters', 'results') else value}"
        for key, value in provenance.items()
        if key not in {"source", "specs"} and not key.endswith("_sha256")
    ]
    logger.kv("Parameters", ", ".join([str(provenance.get("source")), *details]))
    scheduler = config.job_scheduler
    if scheduler == "local" and config.local_max_parallel_jobs > 1:
        scheduler += f", {config.local_max_parallel_jobs} at once"
    if not config.dispatch:
        scheduler += " (stage only)"
    logger.kv("Scheduler", scheduler)
    logger.kv("Stored outputs", ", ".join(config.store) or "none")
    for name, (lower, upper) in specs.bounds.items():
        role = "implicit" if name in specs.implicit_names else "sampled"
        logger.info(f"{name}: [{lower:g}, {upper:g}] {role}", level=2)
    if not config.dispatch:
        logger.warn("Samples are only staged; no MD is run.")
    logger.blank()


def run_campaign(
    config: SimulationCampaignConfig,
    *,
    stage: str,
    title: str,
    specs: Specs,
    draw: ParameterDraw,
) -> None:
    """Stage and run a campaign over parameter samples, or resume one.

    ``draw`` returns the explicit parameter samples and their provenance; it
    is not called when resuming, which reuses the staged samples and runs
    only those not yet completed.
    """
    started = time.perf_counter()
    campaign_dir = config.campaign_dir
    if config.resume:
        prepare_campaign_dir(config)
        samples, provenance = _resumed_samples(config, specs)
    else:
        parameter_samples, provenance = draw()
        parameter_samples = np.asarray(parameter_samples, dtype=float)
        # Checked before an overwrite deletes the previous campaign.
        check_parameter_samples(specs, parameter_samples)
        prepare_campaign_dir(config)
        campaign_dir.mkdir(parents=True, exist_ok=True)
        pad = len(str(len(parameter_samples)))
        samples = {
            f"{index:0{pad}d}": {"params": params.tolist(), "status": "staged"}
            for index, params in enumerate(parameter_samples)
        }
        specs.write(campaign_dir / "specs.yaml")
    pending = [
        sample_id
        for sample_id, sample in samples.items()
        if sample["status"] != "completed"
    ]

    logger = Logger(stage, str(config.log), mode="a" if config.resume else "w")
    logger.section(title)
    _log_summary(
        config,
        logger,
        specs=specs,
        n_samples=len(samples),
        n_pending=len(pending),
        provenance=provenance,
    )

    systems = stage_systems(config.systems, campaign_dir)
    max_hours = None
    if config.job_scheduler == "slurm":
        # Leave 10% of the time limit for setup, minimization, and copying.
        limit = slurm.time_limit_hours((config.slurm.sbatch or {}).get("time"))
        max_hours = None if limit is None else 0.9 * limit
    fn_campaign = campaign_dir / "campaign.yaml"
    save_yaml(
        {
            "gmx_cmd": config.gmx_cmd,
            "store": list(config.store),
            "cleanup": config.cleanup,
            "scratch_dir": config.scratch_dir,
            "max_hours": max_hours,
            "systems": [system.to_dict(campaign_dir) for system in systems],
        },
        fn_campaign,
    )
    for sample_id in pending:
        # Until its job reports otherwise, a dispatched sample has failed.
        samples[sample_id]["status"] = "failed" if config.dispatch else "staged"
    _write_manifest(config, specs, samples, provenance)
    logger.done("Staged", detail=f"{len(pending)} sample(s) in {campaign_dir}")

    fn_script = campaign_dir / "run.sh"
    fn_tasks = campaign_dir / "tasks.txt"
    if config.job_scheduler == "slurm":
        slurm.write_task_script(
            fn_script,
            config=config.slurm,
            sbatch={"output": campaign_dir / "slurm" / "%A_%a.out"},
            commands=[
                f'SAMPLE_ID=$(sed -n "$((TASK_ID + 1))p" {shlex.quote(str(fn_tasks))})',
                f'SAMPLE_DIR={shlex.quote(str(campaign_dir / "samples"))}/"$SAMPLE_ID"',
                'mkdir -p "$SAMPLE_DIR"',
                'exec >>"$SAMPLE_DIR/run.out" 2>&1',
                slurm.bff_command("md", shlex.quote(str(fn_campaign)), '"$SAMPLE_ID"'),
            ],
        )
        fn_tasks.write_text("".join(f"{sample_id}\n" for sample_id in pending))
        (campaign_dir / "slurm").mkdir(exist_ok=True)

    try:
        if not pending:
            logger.info("Every sample is already completed.")
        elif not config.dispatch:
            for sample_id in pending:
                for system in systems:
                    system_dir = campaign_dir / "samples" / sample_id / system.system_id
                    system_dir.mkdir(parents=True, exist_ok=True)
                    write_sample_topology(
                        system.topology_path,
                        specs,
                        samples[sample_id]["params"],
                        system_dir / "topology.top",
                    )
            if config.job_scheduler == "slurm":
                array = slurm.array_option(len(pending), config.slurm.max_parallel_jobs)
                logger.info(f"Submit with: sbatch --array={array} {fn_script}")
        elif config.job_scheduler == "local":
            _forget_results(campaign_dir, pending)
            _run_local(
                fn_campaign,
                pending,
                max_parallel_jobs=config.local_max_parallel_jobs,
                logger=logger,
            )
        else:
            # Samples stopped by the time limit continue from their checkpoints.
            for restart in range(config.max_restarts + 1):
                if restart:
                    logger.info(
                        f"Resubmitting {len(pending)} incomplete sample(s) "
                        f"(restart {restart}/{config.max_restarts})."
                    )
                    fn_tasks.write_text("".join(f"{sid}\n" for sid in pending))
                _forget_results(campaign_dir, pending)
                task_ids = slurm.run_tasks(
                    fn_script, len(pending), config=config.slurm, logger=logger
                )
                for sample_id, task_id in zip(pending, task_ids):
                    samples[sample_id]["job_id"] = task_id
                pending = [
                    sample_id
                    for sample_id in pending
                    if _read_result(campaign_dir, sample_id).get("status")
                    == "incomplete"
                ]
                if not pending:
                    break
    finally:
        # Merge the jobs' result.yaml files into samples.yaml.
        for sample_id, sample in samples.items():
            sample.update(_read_result(campaign_dir, sample_id))
            (campaign_dir / "samples" / sample_id / "result.yaml").unlink(
                missing_ok=True
            )
        _write_manifest(config, specs, samples, provenance)
    if config.dispatch:
        if config.cleanup:
            _cleanup(campaign_dir, samples, systems, config.store)
        if config.compress:
            compress_results(campaign_dir)

    counts = {
        status: sum(sample["status"] == status for sample in samples.values())
        for status in ("completed", "incomplete", "failed", "staged")
    }
    summary = ", ".join(f"{n} {status}" for status, n in counts.items() if n)
    elapsed = time.perf_counter() - started
    if counts["failed"] or counts["incomplete"]:
        logger.warn(f"{summary}; rerun with resume: true to retry the rest.")
    logger.done(title, detail=f"{summary} | {elapsed:.1f} s | {campaign_dir}")
