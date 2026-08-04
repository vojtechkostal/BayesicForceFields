"""Build ID-paired QoI datasets from explicit analysis inputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...domain.sample import SampleSet
from ...io.logs import Logger
from ...io.utils import save_json
from ...qoi.analysis import analyze_input_sets
from ...qoi.data import QoI, QoIDataset
from .config import AnalyzeConfig


def _validate_qoi_blocks(blocks: list[QoI], *, context: str) -> None:
    if not blocks:
        raise ValueError(f"{context} contains no QoI blocks.")
    first = blocks[0]
    for block in blocks[1:]:
        if block.values_per_label != first.values_per_label:
            raise ValueError(
                f"{context}: expected values_per_label={first.values_per_label}, "
                f"got {block.values_per_label}."
            )
        if block.labels != first.labels:
            raise ValueError(
                f"{context}: expected labels {first.labels!r}, got {block.labels!r}."
            )
        if block.n_values != first.n_values:
            raise ValueError(
                f"{context}: expected {first.n_values} values, got {block.n_values}."
            )


def _labels(blocks: list[QoI], system_ids: tuple[str, ...]) -> tuple[str, ...] | None:
    labels = blocks[0].labels
    if len(blocks) == 1:
        return labels
    if labels is None:
        if all(block.n_values == block.values_per_label for block in blocks):
            return system_ids
        return None
    return tuple(
        f"{system_id}:{label}"
        for system_id, block in zip(system_ids, blocks)
        for label in block.labels or ()
    )


def _shared_block_metadata(
    blocks: list[QoI],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = [dict(block.settings) for block in blocks]
    metadata = [dict(block.metadata) for block in blocks]
    shared_settings = (
        settings[0] if all(value == settings[0] for value in settings) else {}
    )
    shared_metadata = (
        metadata[0] if all(value == metadata[0] for value in metadata) else {}
    )
    if not shared_settings:
        shared_metadata["settings_by_system"] = settings
    if metadata and not shared_metadata:
        shared_metadata["metadata_by_system"] = metadata
    return dict(shared_settings), dict(shared_metadata)


def _training_tasks(
    sample_set: SampleSet,
    selected_ids: tuple[str, ...],
) -> list[tuple[str, str, dict[str, Any]]]:
    campaign_ids = [system.system_id for system in sample_set.systems]
    missing = sorted(set(selected_ids) - set(campaign_ids))
    if missing:
        raise ValueError(
            f"training_samples.systems requests IDs absent from "
            f"{sample_set.campaign_dir / 'samples.yaml'}: {missing}."
        )
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for sample in sample_set.samples:
        by_id = dict(zip(sample.system_ids, sample.input_roles))
        for system_id in selected_ids:
            if system_id not in by_id:
                raise ValueError(
                    f"Sample {sample.sample_id!r} is missing system {system_id!r}."
                )
            tasks.append((sample.sample_id, system_id, dict(by_id[system_id])))
    return tasks


def main(fn_config: str | Path) -> None:
    started = time.perf_counter()
    config = AnalyzeConfig.load(fn_config)
    training = config.training_samples
    selected_ids = training.system_ids
    config.output.directory.mkdir(parents=True, exist_ok=True)
    logger = Logger("analyze", str(config.output.log), mode="w")
    sample_set = SampleSet.from_dir(
        training.manifest.parent, manifest=training.manifest
    )
    routines_by_system = {
        system_id: tuple(
            routine for routine in config.routines if system_id in routine.systems
        )
        for system_id in selected_ids
    }
    reference_by_id = {
        system.system_id: system.inputs.inputs for system in config.reference.systems
    }
    reference_tasks = [
        ("reference", system_id, dict(reference_by_id[system_id]))
        for system_id in selected_ids
    ]
    training_tasks = _training_tasks(sample_set, selected_ids)

    logger.section("QoI Analysis")
    logger.kv("Config", config.fn_config)
    logger.kv("Sample manifest", training.manifest)
    logger.kv("Systems", ", ".join(selected_ids))
    logger.kv("Samples", sample_set.n_samples)
    logger.kv("Routines", len(config.routines))
    logger.kv("Output directory", config.output.directory)
    logger.blank()

    ref_frames = config.reference.frames
    reference_results = analyze_input_sets(
        reference_tasks,
        routines_by_system=routines_by_system,
        start=ref_frames.start,
        stop=ref_frames.stop,
        step=ref_frames.step,
        workers=1,
        progress_stride=1,
        progress_label="Reference QoI",
        logger=logger,
        in_memory=config.run.in_memory,
        gc_collect=config.run.gc_collect,
        maxtasksperchild=config.run.maxtasksperchild,
    )
    train_frames = training.frames
    training_results = analyze_input_sets(
        training_tasks,
        routines_by_system=routines_by_system,
        start=train_frames.start,
        stop=train_frames.stop,
        step=train_frames.step,
        workers=training.workers,
        progress_stride=training.progress_stride,
        progress_label="Training QoI",
        logger=logger,
        in_memory=config.run.in_memory,
        gc_collect=config.run.gc_collect,
        maxtasksperchild=config.run.maxtasksperchild,
    )

    sample_ids = sample_set.sample_ids
    raw: dict[str, Any] = {"reference": {}, "samples": {}}
    for routine in config.routines:
        system_ids = routine.systems
        ref_blocks = [
            reference_results["reference"][system_id][routine.name]
            for system_id in system_ids
        ]
        _validate_qoi_blocks(
            ref_blocks, context=f"Reference routine {routine.name!r}"
        )
        sample_rows: list[np.ndarray] = []
        for sample_id in sample_ids:
            blocks = [
                training_results[sample_id][system_id][routine.name]
                for system_id in system_ids
            ]
            _validate_qoi_blocks(
                [*ref_blocks, *blocks],
                context=f"Routine {routine.name!r}, sample {sample_id!r}",
            )
            sample_rows.append(np.concatenate([block.values for block in blocks]))
        settings, metadata = _shared_block_metadata(ref_blocks)
        metadata["system_ids"] = list(system_ids)
        dataset = QoIDataset(
            name=routine.name,
            inputs=sample_set.inputs,
            outputs=np.asarray(sample_rows, dtype=float),
            outputs_ref=np.concatenate([block.values for block in ref_blocks]),
            labels=_labels(ref_blocks, system_ids),
            values_per_label=ref_blocks[0].values_per_label,
            settings=settings,
            metadata=metadata,
        )
        fn_dataset = config.output.directory / f"{routine.name}.pt"
        dataset.write(fn_dataset)
        raw["reference"][routine.name] = [block.to_dict() for block in ref_blocks]
        raw["samples"][routine.name] = {
            sample_id: [
                training_results[sample_id][system_id][routine.name].to_dict()
                for system_id in system_ids
            ]
            for sample_id in sample_ids
        }
        logger.done("QoI dataset", detail=str(fn_dataset), level=1)

    if config.output.write_raw:
        fn_raw = config.output.directory / "raw.json"
        save_json(raw, fn_raw)
        logger.done("Raw QoI data", detail=str(fn_raw), level=1)
    logger.done(
        "Analysis",
        detail=f"finished in {time.perf_counter() - started:.2f}s",
        level=1,
    )
