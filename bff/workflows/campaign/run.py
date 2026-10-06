"""Stage, run, and collect a simulation campaign over parameter samples.

Layout of a campaign directory::

    specs.yaml, samples.yaml, run.sh (Slurm)
    systems/<system_id>/             staged inputs shared by all samples
    samples/<sample_id>/             config.yaml, run.out, gmx.log
    tasks.txt (Slurm)                sample IDs of the current job array
    samples/<sample_id>/<system_id>/ topology and MD outputs
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gmxtopology import Topology

from ... import slurm
from ...domain.samples import write_sample_manifest
from ...domain.specs import Specs
from ...io.logs import Logger
from ...io.utils import compress_results, load_yaml, save_yaml
from .config import SimulationCampaignConfig, SimulationSystemConfig
from .job import write_sample_topology


def _system_record(
    system: SimulationSystemConfig, campaign_dir: Path
) -> dict[str, Any]:
    def relative(path: Path | None) -> str | None:
        return None if path is None else str(path.relative_to(campaign_dir))

    return {
        "system_id": system.system_id,
        "topology": relative(system.topology_path),
        "coordinates": relative(system.coordinates_path),
        "mdp": {
            "em": relative(system.mdp_em_path),
            "prod": relative(system.mdp_production_path),
        },
        "index": relative(system.index_path),
        "bias": relative(system.bias.input_file),
        "n_steps": int(system.n_steps),
    }


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


def log_campaign_summary(
    config: SimulationCampaignConfig,
    n_samples: int,
    logger: Logger,
    *,
    title: str,
) -> None:
    logger.section(title)
    logger.kv("Log file", config.log)
    logger.kv("Campaign directory", config.campaign_dir)
    logger.kv("Systems", len(config.systems))
    logger.kv("Samples", n_samples)
    logger.kv("Scheduler", config.job_scheduler)
    logger.kv("Dispatch", "yes" if config.dispatch else "no (stage only)")
    logger.kv("Stored outputs", ", ".join(config.store) if config.store else "none")
    if not config.dispatch:
        logger.warn("Jobs will only be staged; no MD will be run.")
    if not config.store:
        logger.warn("No simulation outputs are configured to be stored.")


def collect_campaign(
    *,
    samples: dict[str, dict[str, Any]],
    systems: list[SimulationSystemConfig],
    campaign_dir: Path,
    store: tuple[str, ...] = (),
    cleanup: bool = False,
    compress: bool = False,
) -> None:
    """Merge per-sample ``result.yaml`` files into ``samples.yaml``.

    With ``cleanup`` only files whose suffix is listed in ``store`` remain in
    the per-system sample directories.
    """
    records: dict[str, Any] = {}
    for sample_id, sample in samples.items():
        fn_result = campaign_dir / "samples" / sample_id / "result.yaml"
        result = load_yaml(fn_result) if fn_result.is_file() else {}
        fn_result.unlink(missing_ok=True)
        records[sample_id] = {
            "params": sample["params"],
            "job_id": sample.get("job_id"),
            "status": result.get("status", sample["status"]),
            "outputs": result.get("outputs", []),
        }
    write_sample_manifest(
        [_system_record(system, campaign_dir) for system in systems],
        records,
        campaign_dir / "samples.yaml",
    )

    if cleanup:
        keep = {f".{suffix}" for suffix in store}
        for sample_id in samples:
            for system in systems:
                system_dir = campaign_dir / "samples" / sample_id / system.system_id
                for path in system_dir.glob("*"):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.suffix not in keep:
                        path.unlink()
    if compress:
        compress_results(campaign_dir)


def _result_status(campaign_dir: Path, sample_id: str) -> str | None:
    fn_result = campaign_dir / "samples" / sample_id / "result.yaml"
    return load_yaml(fn_result).get("status") if fn_result.is_file() else None


def run_campaign(
    config: SimulationCampaignConfig,
    *,
    fn_specs: Path,
    parameter_samples: np.ndarray,
    logger: Logger,
) -> None:
    """Stage every sample and run it locally or as one Slurm job array."""
    campaign_dir = config.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    systems = stage_systems(config.systems, campaign_dir)
    write_sample_manifest(
        [_system_record(system, campaign_dir) for system in systems],
        {},
        campaign_dir / "samples.yaml",
    )

    n_total = len(parameter_samples)
    pad = len(str(max(n_total, 1)))
    max_hours = None
    if config.job_scheduler == "slurm":
        # Leave 10% of the time limit for setup, minimization, and copying.
        limit = slurm.time_limit_hours((config.slurm.sbatch or {}).get("time"))
        max_hours = None if limit is None else 0.9 * limit
    samples: dict[str, dict[str, Any]] = {}
    for index, params in enumerate(np.asarray(parameter_samples, dtype=float)):
        sample_id = f"{index:0{pad}d}"
        sample_dir = campaign_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(
            {
                "sample_id": sample_id,
                "params": params.tolist(),
                "campaign_dir": str(campaign_dir),
                "fn_specs": str(fn_specs.resolve()),
                "gmx_cmd": config.gmx_cmd,
                "store": list(config.store),
                "cleanup": config.cleanup,
                "scratch_dir": config.scratch_dir,
                "max_hours": max_hours,
                "systems": [system.to_dict() for system in systems],
            },
            sample_dir / "config.yaml",
        )
        samples[sample_id] = {
            "params": params.tolist(),
            "status": "staged" if not config.dispatch else "failed",
        }

    fn_script = campaign_dir / "run.sh"
    fn_tasks = campaign_dir / "tasks.txt"
    if config.job_scheduler == "slurm":
        samples_dir = shlex.quote(str(campaign_dir / "samples"))
        slurm.write_task_script(
            fn_script,
            config=config.slurm,
            sbatch={"output": campaign_dir / "slurm" / "%A_%a.out"},
            commands=[
                f'SAMPLE_ID=$(sed -n "$((TASK_ID + 1))p" {shlex.quote(str(fn_tasks))})',
                f'SAMPLE_DIR={samples_dir}/"$SAMPLE_ID"',
                'exec >>"$SAMPLE_DIR/run.out" 2>&1',
                slurm.bff_command("md", '"$SAMPLE_DIR/config.yaml"'),
            ],
        )
        fn_tasks.write_text("".join(f"{sample_id}\n" for sample_id in samples))
        (campaign_dir / "slurm").mkdir(exist_ok=True)

    action = "Running MD" if config.dispatch else "Staging samples"
    finished = False
    try:
        if not config.dispatch:
            specs = Specs(fn_specs)
            for sample_id, sample in samples.items():
                for system in systems:
                    system_dir = campaign_dir / "samples" / sample_id / system.system_id
                    system_dir.mkdir(parents=True, exist_ok=True)
                    write_sample_topology(
                        system.topology_path,
                        specs,
                        sample["params"],
                        system_dir / "topology.top",
                    )
            if config.job_scheduler == "slurm":
                array = slurm.array_option(n_total, config.slurm.max_parallel_jobs)
                logger.info(f"Submit with: sbatch --array={array} {fn_script}")
        elif config.job_scheduler == "local":
            for index, sample_id in enumerate(samples):
                logger.progress_status(
                    f"{action}: {index:>{pad}d}/{n_total}",
                    index,
                    n_total,
                    overwrite=True,
                    write_file=False,
                )
                sample_dir = campaign_dir / "samples" / sample_id
                with open(sample_dir / "run.out", "w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "bff.cli",
                            "md",
                            str(sample_dir / "config.yaml"),
                        ],
                        cwd=campaign_dir,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    logger.warn(
                        f"Sample {sample_id} failed with exit code "
                        f"{completed.returncode}; see {sample_dir / 'run.out'}."
                    )
        else:
            # Samples stopped by the time limit continue from their checkpoints.
            pending = list(samples)
            for restart in range(config.max_restarts + 1):
                if restart:
                    logger.info(
                        f"Resubmitting {len(pending)} incomplete sample(s) "
                        f"(restart {restart}/{config.max_restarts})."
                    )
                fn_tasks.write_text("".join(f"{sample_id}\n" for sample_id in pending))
                task_ids = slurm.run_tasks(
                    fn_script, len(pending), config=config.slurm, logger=logger
                )
                for sample_id, task_id in zip(pending, task_ids):
                    samples[sample_id]["job_id"] = task_id
                pending = [
                    sample_id
                    for sample_id in pending
                    if _result_status(campaign_dir, sample_id) == "incomplete"
                ]
                if not pending:
                    break
        finished = True
        logger.done(action, detail=f"{n_total}/{n_total}")
    finally:
        collect_campaign(
            samples=samples,
            systems=systems,
            campaign_dir=campaign_dir,
            store=config.store,
            cleanup=config.cleanup and config.dispatch and finished,
            compress=config.compress and config.dispatch and finished,
        )
